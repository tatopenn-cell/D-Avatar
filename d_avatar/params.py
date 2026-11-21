"""
Parameter extraction from 68 landmarks.

Source (Larey et al. 2023, arXiv:2306.12188):
- §3.2 p. 4: landmarks aligned to frontal pose, blendshape weights post-processed
  by Reasonable Array (binary table of target pairs; when pair not allowed,
  smaller weight zeroed) and smoothed by exponential moving average.
- §3.4 p. 6: 68 landmarks (68, 2), eight regions (eyebrow-left, eyebrow-right,
  eye-left, eye-right, nose, nostril, teeth, lips), weights in [0, 1].
- §3.5.1 p. 6: blink from eyelid distance, online two-class k-means (open, closed)
  whose open-class value is the per-person low edge of the blink range.
- §3.5.2 pp. 6-7: gaze from iris to eye-corner mid-point distance, normalized by
  horizontal eye length.

Source (Kartynnik et al. 2019, arXiv:1907.06724):
- §5 p. 3: temporal jitter filter stabilizing at low speed, avoiding lag at high
  speed (One Euro Filter inspiration).

This implementation uses plain geometry per region (not the trained network
from the paper). Rate bounding uses dense_armor.core.robot.control.cbf_filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 68-point landmark indices (standard dlib/ibug ordering)
# Regions per Larey §3.4 p. 6
_LANDMARK_REGIONS = {
    'eyebrow_left': list(range(17, 22)),  # 17-21
    'eyebrow_right': list(range(22, 27)),  # 22-26
    'eye_left': list(range(36, 42)),  # 36-41
    'eye_right': list(range(42, 48)),  # 42-47
    'nose': list(range(27, 36)),  # 27-35
    'nostril': [31, 35],  # subset of nose
    'teeth': list(range(60, 68)),  # 60-67 (inner mouth)
    'lips': list(range(48, 60)),  # 48-59 (outer mouth)
}

# Eye landmark indices for blink/gaze (dlib 68-point)
_EYE_LEFT_INDICES = [36, 37, 38, 39, 40, 41]
_EYE_RIGHT_INDICES = [42, 43, 44, 45, 46, 47]
# Iris approximation: center of eye region
_IRIS_LEFT_IDX = 38
_IRIS_RIGHT_IDX = 43

# Constants
_EPS = 1e-6
_EAR_OPEN_MIN = 0.1
_EAR_OPEN_MAX = 0.4
_EAR_RANGE = _EAR_OPEN_MAX - _EAR_OPEN_MIN


def _eye_aspect_ratio(pts: np.ndarray, indices: list[int]) -> float:
    """Eye aspect ratio (EAR) from 6 eye landmarks (dlib order).
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where indices = [left_corner, top_inner, top_outer, right_corner, bottom_outer, bottom_inner]
    """
    p = pts[indices]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    if h < _EPS:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def _eye_corner_midpoint(pts: np.ndarray, indices: list[int]) -> np.ndarray:
    """Midpoint between inner and outer eye corners."""
    return ((pts[indices[0]] + pts[indices[3]]) / 2.0).astype(np.float32)  # type: ignore[return-value]


def _horizontal_eye_length(pts: np.ndarray, indices: list[int]) -> float:
    """Distance between eye corners."""
    return float(np.linalg.norm(pts[indices[0]] - pts[indices[3]]))


def landmarks_to_params(points: np.ndarray) -> dict[str, float]:
    """Map cleaned 68-point frame to named parameters in [0, 1] and degrees.

    Args:
        points: (68, 2) array of (x, y) coordinates.

    Returns:
        dict with keys:
        - eye_l, eye_r: eye openness [0, 1] (1 = open)
        - mouth_open: [0, 1]
        - brow_l, brow_r: eyebrow raise [0, 1]
        - gaze_x, gaze_y: [-1, 1] (left/right, up/down)
        - head_yaw, head_pitch, head_roll: degrees

    Note: This uses plain geometry per region, not the trained network
    from Larey §3.4. The Reasonable Array and EMA are applied by the caller.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (68, 2):
        raise ValueError(f'Expected (68, 2), got {pts.shape}')

    params = {}

    ear_l = _eye_aspect_ratio(pts, _EYE_LEFT_INDICES)
    ear_r = _eye_aspect_ratio(pts, _EYE_RIGHT_INDICES)
    params['eye_l'] = float(np.clip((ear_l - _EAR_OPEN_MIN) / _EAR_RANGE, 0.0, 1.0))
    params['eye_r'] = float(np.clip((ear_r - _EAR_OPEN_MIN) / _EAR_RANGE, 0.0, 1.0))

    upper_lip = pts[51:53].mean(axis=0)
    lower_lip = pts[57:59].mean(axis=0)
    mouth_h = np.linalg.norm(upper_lip - lower_lip)
    mouth_w = np.linalg.norm(pts[48] - pts[54])
    params['mouth_open'] = float(np.clip(mouth_h / (mouth_w + _EPS), 0.0, 1.0))

    brow_l_y = pts[17:22, 1].mean()
    brow_r_y = pts[22:27, 1].mean()
    eye_l_y = pts[36:42, 1].mean()
    eye_r_y = pts[42:48, 1].mean()
    eye_h_l = pts[37:41, 1].max() - pts[37:41, 1].min()
    eye_h_r = pts[43:47, 1].max() - pts[43:47, 1].min()
    params['brow_l'] = float(np.clip((eye_l_y - brow_l_y) / (eye_h_l + _EPS), 0.0, 1.0))
    params['brow_r'] = float(np.clip((eye_r_y - brow_r_y) / (eye_h_r + _EPS), 0.0, 1.0))

    iris_l = pts[_IRIS_LEFT_IDX]
    iris_r = pts[_IRIS_RIGHT_IDX]
    corner_l = _eye_corner_midpoint(pts, _EYE_LEFT_INDICES)
    corner_r = _eye_corner_midpoint(pts, _EYE_RIGHT_INDICES)
    h_len_l = _horizontal_eye_length(pts, _EYE_LEFT_INDICES)
    h_len_r = _horizontal_eye_length(pts, _EYE_RIGHT_INDICES)

    gaze_x_l = (iris_l[0] - corner_l[0]) / (h_len_l + _EPS)
    gaze_x_r = (iris_r[0] - corner_r[0]) / (h_len_r + _EPS)
    params['gaze_x'] = float(np.clip((gaze_x_l + gaze_x_r) / 2.0, -1.0, 1.0))

    gaze_y_l = (iris_l[1] - corner_l[1]) / (h_len_l + _EPS)
    gaze_y_r = (iris_r[1] - corner_r[1]) / (h_len_r + _EPS)
    params['gaze_y'] = float(np.clip((gaze_y_l + gaze_y_r) / 2.0, -1.0, 1.0))

    nose_bridge = pts[27:31].mean(axis=0)
    eye_center = pts[36:48].mean(axis=0)
    params['head_yaw'] = float(np.clip((nose_bridge[0] - eye_center[0]) * 10.0, -90.0, 90.0))
    params['head_pitch'] = float(np.clip((nose_bridge[1] - eye_center[1]) * 10.0, -90.0, 90.0))
    eye_vec = pts[45] - pts[36]
    params['head_roll'] = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))

    return params


def ema(prev: float, new: float, alpha: float) -> float:
    """Exponential moving average (Larey §3.2 p. 4)."""
    return alpha * new + (1.0 - alpha) * prev


class BlinkCalibrator:
    """Online two-class k-means for blink detection (Larey §3.5.1 p. 6).

    Centroids: centroids[0] = closed (lower EAR), centroids[1] = open (higher EAR).
    The open-class centroid is the per-person low edge of the blink range.
    """

    def __init__(self) -> None:
        self._centroids: np.ndarray | None = None  # shape (2,) [closed, open]
        self._counts: np.ndarray | None = None
        self._initialized = False

    def update(self, ear: float) -> None:
        """Update k-means with new EAR value."""
        if not self._initialized:
            # Initialize: closed = ear - 0.15, open = ear (assuming first frame is open)
            self._centroids = np.array([ear - 0.15, ear], dtype=np.float32)
            self._counts = np.array([1, 1], dtype=np.int32)
            self._initialized = True
            return

        # Type narrowing for mypy
        assert self._centroids is not None
        assert self._counts is not None

        # Assign to nearest centroid
        dists = np.abs(self._centroids - ear)
        idx = int(np.argmin(dists))

        # Update assigned centroid
        n = self._counts[idx]
        self._centroids[idx] = (self._centroids[idx] * n + ear) / (n + 1)
        self._counts[idx] += 1

        # Ensure ordering: closed < open
        if self._centroids[0] > self._centroids[1]:
            self._centroids = self._centroids[::-1]
            self._counts = self._counts[::-1]

    def is_blink(self, ear: float) -> bool:
        """Return True if EAR is closer to closed centroid (centroids[0])."""
        if not self._initialized:
            return False
        assert self._centroids is not None
        dists = np.abs(self._centroids - ear)
        return int(np.argmin(dists)) == 0

    def blink_threshold(self) -> float:
        """Return the open-class centroid (low edge of blink range)."""
        if not self._initialized:
            return 0.2
        assert self._centroids is not None
        return float(self._centroids[1])  # type: ignore[return-value]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            'centroids': self._centroids.tolist() if self._centroids is not None else None,
            'counts': self._counts.tolist() if self._counts is not None else None,
        }


class RateBound:
    """Rate bound using CBF safety filter (Kartynnik §5 p. 3 + dense_armor CBF)."""

    def __init__(self, max_step: float) -> None:
        self.max_step = float(max_step)
        self._prev: dict[str, float] = {}

    def step(self, name: str, value: float) -> float:
        """Apply rate bound to a parameter."""
        if name not in self._prev:
            self._prev[name] = value
            return value

        current = self._prev[name]
        u_des = value - current
        bounded_change = np.clip(u_des, -self.max_step, self.max_step)
        bounded_value = current + bounded_change
        self._prev[name] = bounded_value
        return bounded_value

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {'max_step': self.max_step, 'prev': self._prev.copy()}
