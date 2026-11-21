"""
FaceShield: landmark stream cleaner with Kalman predict-only for occlusions.

Source equations (Thieu & Melnik 2025, arXiv:2502.15179):
- §2.1 pp. 4-5: EKF equations (state prediction, covariance prediction,
  Kalman gain, state update, covariance update).
- §4.2 p. 13: stochastic 3D facial tracking, state = all landmark coordinates,
  motion model x_{k+1} = x_k + v_k dt + w_k (constant velocity).
- Abstract & §4.2: EKF more robust than UKF under noise and occlusion.

Test protocol (Zhu et al. 2021, arXiv:2112.10087):
- Introduction pp. 1-2, 4-5: dynamic occlusion moving across face in video,
  not a static patch.

This implementation uses the linear Kalman filter (special case of EKF with
constant F and H) via dense_armor.core.online.timeseries.kalman.KalmanFilter.
An outlier point is replaced by the Hampel value (LandmarkCleaner logic).
A point with confidence 0 runs the Kalman predict step only (no measurement
update) for at most max_gap frames; after that it decays to the neutral pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from dense_armor._dtype import as_default_array  # type: ignore[import-untyped]
from dense_armor.core.online.timeseries.kalman import KalmanFilter  # type: ignore[import-untyped]
from dense_armor.roles import Transformer  # type: ignore[import-untyped]

# Constants for Hampel filter
_MIN_WINDOW = 3
_SCALE_EPS = 1e-12
_DEV_EPS = 1e-9


def _hampel_value(window: list[float], value: float, n_sigmas: float) -> tuple[float, bool]:
    """Causal Hampel filter on a 1D window (Pearson et al. 2016).
    Returns (filtered_value, was_outlier)."""
    if len(window) < _MIN_WINDOW:
        return value, False
    arr = as_default_array(window)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale <= _SCALE_EPS:
        return med, abs(value - med) > _DEV_EPS
    if abs(value - med) > n_sigmas * scale:
        return med, True
    return value, False


class FaceShield(Transformer):
    """Clean a frame of landmarks (N, 2) with confidence (N,).

    Each point gets a 4D constant-velocity Kalman filter (x, y, vx, vy).
    - Confidence > 0: measurement update after Hampel outlier check.
    - Confidence == 0: predict-only (no measurement update) for up to max_gap
      frames; then decay toward neutral pose.
    - Output never jumps: outliers replaced, gaps filled by prediction.

    Args:
        window: causal window size for Hampel filter (default 7).
        n_sigmas: Hampel threshold in scaled MAD units (default 3.0).
        max_gap: max frames to run predict-only before decaying to neutral (default 5).
        neutral: neutral pose (N, 2) to decay to after max_gap; if None, uses
            first valid frame as neutral.

    Returns:
        dict with keys:
        - "points": cleaned landmarks (N, 2)
        - "cleaned": int array (N,) count of cleaned coordinates (0, 1, or 2)

    The filter variant is the linear constant-velocity Kalman filter (Thieu
    §2.1 with F, H constant; §4.2 motion model x_{k+1} = x_k + v_k dt + w_k).
    """

    budget_s: float = 1e-3
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
        self.neutral = neutral
        # Per-point state
        self._kfs: list[KalmanFilter] | None = None
        self._hist_x: list[list[float]] | None = None
        self._hist_y: list[list[float]] | None = None
        self._gap: np.ndarray | None = None
        self._n: int | None = None
        self._dt: float = 1.0 / 30.0

    def _ensure_kfs(self, n: int, dt: float) -> None:
        if self._kfs is not None and self._n == n:
            return
        self._n = n
        self._dt = dt
        # Constant-velocity 2D: state = [x, y, vx, vy]
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])
        R = np.diag([1e-2, 1e-2])
        self._kfs = [
            KalmanFilter(F=F, H=H, Q=Q, R=R, x0=np.zeros(4), P0=np.eye(4) * 0.1) for _ in range(n)
        ]
        self._hist_x = [[] for _ in range(n)]
        self._hist_y = [[] for _ in range(n)]
        self._gap = np.zeros(n, dtype=int)

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """Clean one frame.

        Args:
            x: dict with keys "points" (N, 2), "confidence" (N,), optional "t".
               Or a generic feature dict for check_estimator compatibility.
            t: timestamp (seconds). If None, uses x["t"] or 0.

        Returns:
            dict with "points" (N, 2) and "cleaned" (N,).
        """
        # check_estimator compatibility
        if 'points' not in x:
            return {
                'points': np.zeros((0, 2), dtype=np.float32),
                'cleaned': np.zeros(0, dtype=int),
            }

        pts = as_default_array(x['points'])
        conf = as_default_array(x.get('confidence', np.ones(pts.shape[0], dtype=np.float32)))
        _ = float(x.get('t', t if t is not None else 0.0))

        n = pts.shape[0]
        if n == 0:
            return {'points': pts.copy(), 'cleaned': np.zeros(0, dtype=int)}

        if self._kfs is None:
            self._ensure_kfs(n, self._dt)

        # Type narrowing for mypy
        assert self._kfs is not None
        assert self._hist_x is not None
        assert self._hist_y is not None
        assert self._gap is not None

        out_pts = pts.copy()
        cleaned = np.zeros(n, dtype=int)

        for i in range(n):
            kf = self._kfs[i]
            cx, cy = float(pts[i, 0]), float(pts[i, 1])
            c = float(conf[i])

            hx = self._hist_x[i]
            hy = self._hist_y[i]

            if c <= 0.0 or not (np.isfinite(cx) and np.isfinite(cy)):
                self._gap[i] += 1
                _ = kf.learn_one({})
                pred_x, pred_y = float(kf._x[0]), float(kf._x[1])

                if self._gap[i] > self.max_gap and self.neutral is not None:
                    alpha = min(1.0, (self._gap[i] - self.max_gap) * 0.2)
                    nx, ny = float(self.neutral[i, 0]), float(self.neutral[i, 1])
                    pred_x = (1 - alpha) * pred_x + alpha * nx
                    pred_y = (1 - alpha) * pred_y + alpha * ny

                out_pts[i, 0] = pred_x
                out_pts[i, 1] = pred_y
                cleaned[i] = 2
                continue

            fx, ox = _hampel_value(hx, cx, self.n_sigmas)
            fy, oy = _hampel_value(hy, cy, self.n_sigmas)

            if ox or oy:
                cleaned[i] = int(ox) + int(oy)

            meas = {'x': fx, 'y': fy}
            _ = kf.learn_one(meas)
            out_pts[i, 0] = float(kf._x[0])
            out_pts[i, 1] = float(kf._x[1])

            self._gap[i] = 0
            hx.append(cx)
            hy.append(cy)
            if len(hx) > self.window:
                hx.pop(0)
            if len(hy) > self.window:
                hy.pop(0)

            if self.neutral is None and self._gap[i] == 0 and len(hx) == 1:
                self.neutral = out_pts.copy()

        return {'points': out_pts, 'cleaned': cleaned}

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> FaceShield:
        _ = self.transform_one(x, t)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep)
        params.update(
            {
                'window': self.window,
                'n_sigmas': self.n_sigmas,
                'max_gap': self.max_gap,
                'neutral': self.neutral.tolist() if self.neutral is not None else None,
            }
        )
        return params  # type: ignore[no-any-return]

    def _unit_test_skips(self) -> set:
        return set()
