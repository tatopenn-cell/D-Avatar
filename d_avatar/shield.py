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
The outlier test is the causal Hampel rule of dense_armor LandmarkCleaner.
A point with confidence 0 runs the Kalman predict step only (no measurement
update) for up to max_gap frames; after that it decays to the neutral pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from dense_armor._dtype import as_default_array  # type: ignore[import-untyped]
from dense_armor.core.online.timeseries.kalman import KalmanFilter  # type: ignore[import-untyped]
from dense_armor.roles import Transformer  # type: ignore[import-untyped]

_MIN_WINDOW = 3
_SCALE_EPS = 1e-12
_DEV_EPS = 1e-9
_DT_EPS = 1e-6
_DECAY_STEP = 0.2


def _hampel_value(window: list[float], value: float, n_sigmas: float) -> tuple[float, bool]:
    """Causal Hampel filter on a 1D window, following dense_armor LandmarkCleaner.

    Returns (filtered_value, was_outlier).
    """
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
        self.neutral = np.asarray(neutral, dtype=float) if neutral is not None else None
        self._neutral_given = neutral is not None
        self._kfs: list[KalmanFilter] | None = None
        self._hist_x: list[list[float]] | None = None
        self._hist_y: list[list[float]] | None = None
        self._gap: np.ndarray | None = None
        self._initialized: np.ndarray | None = None
        self._neutral_valid: np.ndarray | None = None
        self._n: int | None = None
        self._dt: float = 1.0 / 30.0
        self._last_t: float | None = None

    def _ensure_kfs(self, n: int) -> None:
        if self._kfs is not None and self._n == n:
            return
        self._n = n
        self._dt = 1.0 / 30.0
        self._last_t = None
        dt = self._dt
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        Q = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
        R = np.diag([1e-3, 1e-3])
        self._kfs = [
            KalmanFilter(F=F.copy(), H=H, Q=Q, R=R, x0=np.zeros(4), P0=np.eye(4) * 0.1)
            for _ in range(n)
        ]
        self._hist_x = [[] for _ in range(n)]
        self._hist_y = [[] for _ in range(n)]
        self._gap = np.zeros(n, dtype=int)
        self._initialized = np.zeros(n, dtype=bool)
        if self._neutral_given and self.neutral is not None:
            self._neutral_valid = np.ones(n, dtype=bool)
        else:
            self.neutral = np.full((n, 2), np.nan, dtype=float)
            self._neutral_valid = np.zeros(n, dtype=bool)

    def _step_time(self, current_t: float | None) -> None:
        assert self._kfs is not None
        dt = self._dt
        if current_t is not None:
            if self._last_t is not None and current_t - self._last_t > 0:
                dt = current_t - self._last_t
            self._last_t = current_t
        if abs(dt - self._dt) > _DT_EPS:
            self._dt = dt
            for kf in self._kfs:
                kf.F[0, 2] = dt
                kf.F[1, 3] = dt

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

    def _occluded(self, i: int, pt: np.ndarray, out: np.ndarray, cleaned: np.ndarray) -> None:
        assert self._kfs is not None
        assert self._gap is not None
        assert self._initialized is not None
        assert self._neutral_valid is not None
        assert self.neutral is not None
        cleaned[i] = 2
        if not self._initialized[i]:
            out[i] = pt if np.all(np.isfinite(pt)) else 0.0
            return
        self._gap[i] += 1
        kf = self._kfs[i]
        kf.learn_one(None)
        assert kf._x is not None
        px, py = float(kf._x[0]), float(kf._x[1])
        if self._gap[i] > self.max_gap and self._neutral_valid[i]:
            a = min(1.0, (self._gap[i] - self.max_gap) * _DECAY_STEP)
            px = (1 - a) * px + a * float(self.neutral[i, 0])
            py = (1 - a) * py + a * float(self.neutral[i, 1])
        out[i, 0] = px
        out[i, 1] = py

    def _measured(self, i: int, pt: np.ndarray, out: np.ndarray, cleaned: np.ndarray) -> None:
        assert self._kfs is not None
        assert self._hist_x is not None
        assert self._hist_y is not None
        assert self._gap is not None
        assert self._initialized is not None
        kf = self._kfs[i]
        cx, cy = float(pt[0]), float(pt[1])
        hx, hy = self._hist_x[i], self._hist_y[i]
        if self._gap[i] > 0:
            hx.clear()
            hy.clear()
        if not self._initialized[i]:
            kf.predict_one()
            assert kf._x is not None
            assert kf._P is not None
            kf._x[:] = [cx, cy, 0.0, 0.0]
            kf._P[:, :] = np.diag([1e-6, 1e-6, 1e-2, 1e-2])
            self._initialized[i] = True
            out[i] = (cx, cy)
        else:
            fx, ox = _hampel_value(hx, cx, self.n_sigmas)
            fy, oy = _hampel_value(hy, cy, self.n_sigmas)
            cleaned[i] = int(ox) + int(oy)
            kf.learn_one({'x': fx, 'y': fy})
            assert kf._x is not None
            out[i, 0] = float(kf._x[0])
            out[i, 1] = float(kf._x[1])
        self._gap[i] = 0
        hx.append(cx)
        hy.append(cy)
        if len(hx) > self.window:
            hx.pop(0)
            hy.pop(0)

    def _learn_neutral(self, out: np.ndarray, measured: np.ndarray) -> None:
        if self._neutral_given:
            return
        assert self._neutral_valid is not None
        assert self.neutral is not None
        fresh = measured & ~self._neutral_valid
        self.neutral[fresh] = out[fresh]
        self._neutral_valid[fresh] = True

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
        self._ensure_kfs(n)
        self._step_time(frame_t if frame_t is not None else t)
        out = pts.astype(float).copy()
        cleaned = np.zeros(n, dtype=int)
        measured = np.zeros(n, dtype=bool)
        for i in range(n):
            ok = float(conf[i]) > 0.0 and bool(np.all(np.isfinite(pts[i])))
            if ok:
                self._measured(i, pts[i], out, cleaned)
                measured[i] = True
            else:
                self._occluded(i, pts[i], out, cleaned)
        self._learn_neutral(out, measured)
        return {'points': out, 'cleaned': cleaned}

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
