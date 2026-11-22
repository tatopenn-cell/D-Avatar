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

This implementation is the linear Kalman filter (special case of EKF with constant F and H),
written as batched numpy operations over all points. It follows the equations of
dense_armor.core.online.timeseries.kalman.KalmanFilter, which the tests use as reference.
The outlier test is the causal Hampel rule of dense_armor LandmarkCleaner.
A point with confidence 0 runs the Kalman predict step only (no measurement
update) for up to max_gap frames; after that it decays to the neutral pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from dense_armor._dtype import as_default_array  # type: ignore[import-untyped]
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


def _hampel_batch(
    hist: np.ndarray, vals: np.ndarray, n_sigmas: float
) -> tuple[np.ndarray, np.ndarray]:
    """Hampel rule of ``_hampel_value`` for N points at once.

    Args:
        hist: (N, W) history, left-padded with NaN.
        vals: (N,) new values.
        n_sigmas: threshold in scaled MAD units.

    Returns:
        (filtered values (N,), outlier flags (N,))
    """
    out: np.ndarray = np.zeros(len(vals), dtype=bool)
    filt = vals.copy()
    rows = np.sum(~np.isnan(hist), axis=1) >= _MIN_WINDOW
    if rows.any():
        h = hist[rows]
        med = np.nanmedian(h, axis=1)
        scale = 1.4826 * np.nanmedian(np.abs(h - med[:, None]), axis=1)
        dev = np.abs(vals[rows] - med)
        flag = np.where(scale <= _SCALE_EPS, dev > _DEV_EPS, dev > n_sigmas * scale)
        out[rows] = flag
        filt[rows] = np.where(flag, med, vals[rows])
    return filt, out


_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_Q = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
_R = np.diag([1e-3, 1e-3])
_P_INIT = np.diag([1e-6, 1e-6, 1e-2, 1e-2])


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
        self._x: np.ndarray | None = None
        self._p: np.ndarray | None = None
        self._hist: np.ndarray | None = None
        self._gap: np.ndarray | None = None
        self._initialized: np.ndarray | None = None
        self._f = np.eye(4)
        self._neutral_valid: np.ndarray | None = None
        self._n: int | None = None
        self._dt: float = 1.0 / 30.0
        self._last_t: float | None = None

    def _ensure_state(self, n: int) -> None:
        if self._x is not None and self._n == n:
            return
        self._n = n
        self._dt = 1.0 / 30.0
        self._last_t = None
        self._f = np.eye(4)
        self._f[0, 2] = self._f[1, 3] = self._dt
        self._x = np.zeros((n, 4))
        self._p = np.tile(_P_INIT, (n, 1, 1))
        self._hist = np.full((n, 2, self.window), np.nan)
        self._gap = np.zeros(n, dtype=int)
        self._initialized = np.zeros(n, dtype=bool)
        if self._neutral_given and self.neutral is not None:
            self._neutral_valid = np.ones(n, dtype=bool)
        else:
            self.neutral = np.full((n, 2), np.nan, dtype=float)
            self._neutral_valid = np.zeros(n, dtype=bool)

    def _step_time(self, current_t: float | None) -> None:
        dt = self._dt
        if current_t is not None:
            if self._last_t is not None and current_t - self._last_t > 0:
                dt = current_t - self._last_t
            self._last_t = current_t
        if abs(dt - self._dt) > _DT_EPS:
            self._dt = dt
            self._f[0, 2] = self._f[1, 3] = dt

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

    def _update(self, pts: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._x is not None
        assert self._p is not None
        assert self._hist is not None
        assert self._gap is not None
        assert self._initialized is not None
        assert self._neutral_valid is not None
        assert self.neutral is not None
        n = len(pts)
        out = np.where(np.isfinite(pts), pts, 0.0).astype(float)
        cleaned = np.where(ok, 0, 2)
        xp = self._x @ self._f.T
        pp = self._f @ self._p @ self._f.T + _Q
        init = self._initialized

        occ = ~ok & init
        self._x[occ] = xp[occ]
        self._p[occ] = pp[occ]
        self._gap[occ] += 1
        out[occ] = xp[occ, :2]
        decay = occ & (self._gap > self.max_gap) & self._neutral_valid
        a = np.minimum(1.0, (self._gap[decay] - self.max_gap) * _DECAY_STEP)[:, None]
        out[decay] = (1 - a) * out[decay] + a * self.neutral[decay]

        self._hist[ok & (self._gap > 0)] = np.nan
        upd = ok & init
        if upd.any():
            fx, ox = _hampel_batch(self._hist[upd, 0], pts[upd, 0], self.n_sigmas)
            fy, oy = _hampel_batch(self._hist[upd, 1], pts[upd, 1], self.n_sigmas)
            cleaned[upd] = ox.astype(int) + oy.astype(int)
            pu = pp[upd]
            s_inv = np.linalg.inv(pu[:, :2, :2] + _R)
            gain = pu[:, :, :2] @ s_inv
            innov = np.stack([fx, fy], axis=1) - xp[upd, :2]
            self._x[upd] = xp[upd] + (gain @ innov[..., None])[..., 0]
            p_new = (np.eye(4) - gain @ _H) @ pu
            self._p[upd] = 0.5 * (p_new + p_new.transpose(0, 2, 1))
            out[upd] = self._x[upd, :2]
        new = ok & ~init
        if new.any():
            self._x[new] = np.concatenate([pts[new], np.zeros((int(new.sum()), 2))], axis=1)
            self._p[new] = _P_INIT
            self._initialized[new] = True
            out[new] = pts[new]
        self._gap[ok] = 0
        self._hist[ok, :, :-1] = self._hist[ok, :, 1:]
        self._hist[ok, :, -1] = pts[ok]
        assert len(out) == n
        return out, cleaned

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
        self._ensure_state(n)
        self._step_time(frame_t if frame_t is not None else t)
        measured = (conf > 0.0) & np.all(np.isfinite(pts), axis=1)
        out, cleaned = self._update(pts.astype(float), measured)
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
