"""
FaceShield: landmark stream cleaner with Kalman predict-only for occlusions.

This is a thin wrapper over Dense-Armor's BatchedLandmarkShield, which implements
a batched constant-velocity Kalman filter with causal Hampel outlier filter per
coordinate. See dense_armor.core.signals.vision.landmarks.BatchedLandmarkShield
for the full documentation.

Source equations (Thieu & Melnik 2025, arXiv:2502.15179):
- §2.1 pp. 4-5: EKF equations (state prediction, covariance prediction,
  Kalman gain, state update, covariance update).
- §4.2 p. 13: stochastic 3D facial tracking, state = all landmark coordinates,
  motion model x_{k+1} = x_k + v_k dt + w_k (constant velocity).
- Abstract & §4.2: EKF more robust than UKF under noise and occlusion.

Test protocol (Zhu et al. 2021, arXiv:2112.10087):
- Introduction pp. 1-2, 4-5: dynamic occlusion moving across face in video,
  not a static patch.

This wrapper adds timestamp handling (dt from frame timestamps) and flat-dict
input support on top of Dense-Armor's BatchedLandmarkShield.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from dense_armor._dtype import as_default_array  # type: ignore[import-untyped]
from dense_armor.core.signals.vision.landmarks import (  # type: ignore[import-untyped]
    _DT_EPS,
    BatchedLandmarkShield,
    _hampel_batch,
    _hampel_value,
)
from dense_armor.roles import Transformer  # type: ignore[import-untyped]

# _DT_EPS is imported from dense_armor, don't redefine

# Re-export for tests and external use
__all__ = ['FaceShield', '_hampel_batch', '_hampel_value']


class FaceShield(Transformer):
    """Clean a frame of landmarks (N, 2) with confidence (N,).

    Thin wrapper over ``dense_armor.core.signals.vision.landmarks.BatchedLandmarkShield``.
    Adds timestamp-based dt calculation and flat-dict input support.

    Each point gets a 4D constant-velocity Kalman filter (x, y, vx, vy).
    - Confidence > 0: measurement update after the Hampel outlier check. The
      first valid measurement of a point initialises its filter at that
      measurement, with zero velocity.
    - Confidence == 0: predict-only (no measurement update). A point that was
      never measured is returned as given. After max_gap frames of occlusion the
      prediction decays toward the neutral pose of that point.
    - The Hampel history of a point is cleared when the point becomes valid
      again after an occlusion.
    - The time step comes from the frame timestamps; the first frame uses 1/30 s.

    Args:
        window: causal window size for Hampel filter (default 7).
        n_sigmas: Hampel threshold in scaled MAD units (default 3.0).
        max_gap: max frames to run predict-only before decaying to neutral (default 5).
        neutral: neutral pose (N, 2) to decay to after max_gap. If None, the
            neutral of each point is its first valid measurement; it is undefined
            (NaN in ``neutral``) until the point has been measured once, and
            the decay does not apply to it.

    Input:
        A frame ``{"points": (N, 2), "confidence": (N,), "t": float}``, or a
        flat dict of numbers ``{"a": 1.0, "b": 2.0}``, read as N points on the
        x axis (y = 0, confidence 1) in sorted key order.

    Returns:
        dict with keys:
        - "points": cleaned landmarks (N, 2)
        - "cleaned": int array (N,) count of cleaned coordinates (0, 1, or 2)

    The filter variant is the linear constant-velocity Kalman filter (Thieu
    §2.1 with F, H constant; §4.2 motion model x_{k+1} = x_k + v_k dt + w_k).
    """

    budget_s: float = 3.5e-2
    memory_class: str = 'O(N * window)'

    def __init__(
        self,
        window: int = 7,
        n_sigmas: float = 3.0,
        max_gap: int = 5,
        neutral: np.ndarray | None = None,
    ) -> None:
        self.window = int(window)
        self.n_sigmas = float(n_sigmas)
        self.max_gap = int(max_gap)
        self.neutral = np.asarray(neutral, dtype=float) if neutral is not None else None
        self._neutral_given = neutral is not None

        # Wrapped BatchedLandmarkShield
        self._shield = BatchedLandmarkShield(
            window=self.window,
            n_sigmas=self.n_sigmas,
            max_gap=self.max_gap,
            neutral=self.neutral,
        )

        # Timestamp handling
        self._dt: float = 1.0 / 30.0
        self._last_t: float | None = None

    def _step_time(self, current_t: float | None) -> None:
        dt = self._dt
        if current_t is not None:
            if self._last_t is not None and current_t - self._last_t > 0:
                dt = current_t - self._last_t
            self._last_t = current_t
        if abs(dt - self._dt) > _DT_EPS:
            self._dt = dt

    def _frame(self, x: dict) -> tuple[np.ndarray, np.ndarray, float | None]:
        if 'points' in x:
            pts = as_default_array(x['points'])
            conf = as_default_array(x.get('confidence', np.ones(pts.shape[0])))
            t = x.get('t')
            return pts, conf, None if t is None else float(t)
        keys = sorted(k for k in x if k != 't')
        vals = np.array([float(x[k]) for k in keys], dtype=float)
        pts = np.stack([vals, np.zeros_like(vals)], axis=1) if keys else np.zeros((0, 2))
        t = x.get('t')
        return pts, np.ones(len(keys)), None if t is None else float(t)

    def _learn_neutral(self, out: np.ndarray, measured: np.ndarray) -> None:
        if self._neutral_given:
            return
        # Delegate to wrapped shield
        self._shield._learn_neutral(out, measured)
        # Sync neutral attribute from wrapped shield
        self.neutral = self._shield.neutral

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """Clean one frame.

        Args:
            x: a frame dict ("points" (N, 2), "confidence" (N,), optional "t"),
                or a flat dict of numbers read as N points on the x axis.
            t: timestamp in seconds, used when the frame has no "t".

        Returns:
            dict with "points" (N, 2) and "cleaned" (N,).
        """
        pts, conf, frame_t = self._frame(x)
        n = pts.shape[0]
        if n == 0:
            return {'points': pts.copy(), 'cleaned': np.zeros(0, dtype=int)}

        # Update dt from timestamp
        self._step_time(frame_t if frame_t is not None else t)

        # Update wrapped shield's dt if it changed
        if abs(self._dt - self._shield._dt) > _DT_EPS:
            self._shield._dt = self._dt
            self._shield._f[0, 2] = self._shield._f[1, 3] = self._dt

        # Delegate to wrapped shield
        frame_dict: dict[str, Any] = {'points': pts, 'confidence': conf}
        if frame_t is not None:
            frame_dict['t'] = frame_t
        elif t is not None:
            frame_dict['t'] = t

        result: dict = self._shield.transform_one(frame_dict)

        # Learn neutral from wrapped shield
        measured = (conf > 0.0) & np.all(np.isfinite(pts), axis=1)
        self._learn_neutral(result['points'], measured)

        return result

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> FaceShield:
        self.transform_one(x, t)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep)
        params.update(
            {
                'window': self.window,
                'n_sigmas': self.n_sigmas,
                'max_gap': self.max_gap,
                'neutral': self.neutral.tolist()
                if self._neutral_given and self.neutral is not None
                else None,
            }
        )
        return params  # type: ignore[no-any-return]

    def _unit_test_skips(self) -> set:
        return set()
