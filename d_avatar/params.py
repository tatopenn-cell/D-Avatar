"""
Parameter extraction from 68 landmarks.

Source (Larey et al. 2023, arXiv:2306.12188):
- §3.2 p. 4: blendshape weights smoothed over frames by an exponential moving average.
- §3.4 p. 6: 68 landmarks (68, 2), weights in [0, 1].
- §3.5.1 p. 6: blink from the eyelid distance, online two-class k-means (open, closed);
  the open-class value is the per-person scale of the eye opening.

The mapping is plain geometry on the iBUG ordering, not the trained network of the paper.
The eyebrow, mouth and head-angle scalings are geometric heuristics with fixed constants
(see the module constants); they are not taken from a paper and are not calibrated.
Gaze is not computed: the 68-point layout has no iris landmarks (Larey §3.5.2 needs them).
"""

from __future__ import annotations

from typing import Any

import numpy as np

EAR_OPEN_DEFAULT = 0.3
_EPS = 1e-6
_BROW_MIN = 1.0
_BROW_MAX = 2.0
_MOUTH_MAX = 0.5
_DEG_PER_IOD = 60.0
_CLOSED_RATIO = 0.3
_EYE_L = (36, 37, 38, 39, 40, 41)
_EYE_R = (42, 43, 44, 45, 46, 47)


def eye_aspect_ratio(pts: np.ndarray, idx: tuple[int, ...]) -> float:
    """Eye aspect ratio from 6 eye landmarks p1..p6 in iBUG order.

    EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)
    """
    p = pts[list(idx)]
    h = float(np.linalg.norm(p[0] - p[3]))
    if h < _EPS:
        return 0.0
    v = float(np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
    return v / (2.0 * h)


def eye_aspect_ratios(points: np.ndarray) -> tuple[float, float]:
    """EAR of the left and right eye of a (68, 2) frame."""
    pts = np.asarray(points, dtype=float)
    return eye_aspect_ratio(pts, _EYE_L), eye_aspect_ratio(pts, _EYE_R)


def landmarks_to_params(points: np.ndarray, ear_open: float = EAR_OPEN_DEFAULT) -> dict[str, float]:
    """Map a cleaned 68-point frame to named parameters.

    Args:
        points: (68, 2) array of (x, y) coordinates.
        ear_open: eye aspect ratio of this person with open eyes (see BlinkCalibrator).

    Returns:
        dict with
        - eye_l, eye_r: min(EAR / ear_open, 1), 1 = open
        - mouth_open: lip gap over mouth width, divided by 0.5, in [0, 1]
        - brow_l, brow_r: eyebrow-to-eye distance over eye width, mapped from [1, 2] to [0, 1]
        - head_yaw, head_pitch: nose-bridge offset from the eye centre over the
          inter-ocular distance, times 60 degrees, in [-90, 90]
        - head_roll: angle of the line between the outer eye corners, in degrees

    Raises:
        ValueError: if points is not (68, 2).
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape != (68, 2):
        raise ValueError(f'Expected (68, 2), got {pts.shape}')
    ear_l, ear_r = eye_aspect_ratios(pts)
    scale = max(float(ear_open), _EPS)
    params = {
        'eye_l': float(np.clip(ear_l / scale, 0.0, 1.0)),
        'eye_r': float(np.clip(ear_r / scale, 0.0, 1.0)),
    }
    mouth_w = float(np.linalg.norm(pts[48] - pts[54]))
    mouth_h = float(np.linalg.norm(pts[51:53].mean(axis=0) - pts[57:59].mean(axis=0)))
    params['mouth_open'] = float(np.clip(mouth_h / (mouth_w + _EPS) / _MOUTH_MAX, 0.0, 1.0))
    for name, brow, eye in (('brow_l', slice(17, 22), _EYE_L), ('brow_r', slice(22, 27), _EYE_R)):
        eye_w = float(np.linalg.norm(pts[eye[0]] - pts[eye[3]]))
        rise = float(pts[list(eye), 1].mean() - pts[brow, 1].mean()) / (eye_w + _EPS)
        params[name] = float(np.clip((rise - _BROW_MIN) / (_BROW_MAX - _BROW_MIN), 0.0, 1.0))
    eye_l_c = pts[list(_EYE_L)].mean(axis=0)
    eye_r_c = pts[list(_EYE_R)].mean(axis=0)
    iod = float(np.linalg.norm(eye_r_c - eye_l_c)) + _EPS
    offset = pts[27:31].mean(axis=0) - (eye_l_c + eye_r_c) / 2.0
    params['head_yaw'] = float(np.clip(offset[0] / iod * _DEG_PER_IOD, -90.0, 90.0))
    params['head_pitch'] = float(np.clip(offset[1] / iod * _DEG_PER_IOD, -90.0, 90.0))
    corner = pts[45] - pts[36]
    params['head_roll'] = float(np.degrees(np.arctan2(corner[1], corner[0])))
    return params


def ema(prev: float, new: float, alpha: float) -> float:
    """Exponential moving average (Larey §3.2 p. 4)."""
    return alpha * new + (1.0 - alpha) * prev


class BlinkCalibrator:
    """Online two-class k-means on the eye aspect ratio (Larey §3.5.1 p. 6).

    The classes are closed (index 0) and open (index 1). The first sample seeds the open
    centroid, and the closed centroid is seeded at 0.3 times that value, so the first
    sample is assumed to be an open eye. Each sample updates its nearest centroid with a
    running mean whose count is capped at ``max_count``, so the centroids keep adapting.

    Args:
        max_count: cap of the per-centroid count (default 200).
    """

    def __init__(self, max_count: int = 200) -> None:
        self.max_count = int(max_count)
        self._c: np.ndarray | None = None
        self._n: np.ndarray | None = None

    def update(self, ear: float) -> None:
        """Update the centroids with one EAR value."""
        if self._c is None or self._n is None:
            self._c = np.array([ear * _CLOSED_RATIO, ear], dtype=float)
            self._n = np.array([1, 1], dtype=int)
            return
        k = int(np.argmin(np.abs(self._c - ear)))
        self._n[k] = min(int(self._n[k]) + 1, self.max_count)
        self._c[k] += (ear - self._c[k]) / self._n[k]

    def open_ear(self) -> float:
        """EAR of the open class; EAR_OPEN_DEFAULT before the first sample."""
        return EAR_OPEN_DEFAULT if self._c is None else float(self._c[1])

    def is_blink(self, ear: float) -> bool:
        """True if the EAR is closer to the closed centroid."""
        if self._c is None:
            return False
        return int(np.argmin(np.abs(self._c - ear))) == 0

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            'max_count': self.max_count,
            'centroids': None if self._c is None else self._c.tolist(),
        }


class RateBound:
    """Clip the per-frame change of each named parameter to a maximum step.

    Args:
        max_step: maximum absolute change per frame, for every name without its own bound.
        per_name: maximum change per frame for specific names (e.g. head angles in degrees).

    The first value of a name is returned unchanged.
    """

    def __init__(self, max_step: float, per_name: dict[str, float] | None = None) -> None:
        self.max_step = float(max_step)
        self.per_name = dict(per_name) if per_name else {}
        self._prev: dict[str, float] = {}

    def step(self, name: str, value: float) -> float:
        """Return the value moved at most the bound of ``name`` away from the previous output."""
        prev = self._prev.get(name)
        if prev is None:
            self._prev[name] = float(value)
            return float(value)
        bound = self.per_name.get(name, self.max_step)
        out = prev + float(np.clip(value - prev, -bound, bound))
        self._prev[name] = out
        return out

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {'max_step': self.max_step, 'per_name': dict(self.per_name)}
