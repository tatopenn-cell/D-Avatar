"""Tests for the 2D avatar drawer (draw_avatar): pure function, no camera."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from d_avatar.avatar import draw_avatar


def _img(params: dict[str, Any] | None = None, size: int = 400) -> np.ndarray:
    return draw_avatar(params, size=size)


def _count_white_in_eye(img: np.ndarray, side: str = 'left') -> int:
    """Count near-white pixels inside the left or right eye ROI."""
    h, w = img.shape[:2]
    cx, cy = w // 2, int(h * 0.525)
    if side == 'left':
        x0, x1 = cx - 80, cx - 20
    else:
        x0, x1 = cx + 20, cx + 80
    y0, y1 = cy - 40, cy + 10
    roi = img[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
    return int(np.sum(np.all(roi > 200, axis=2)))


def _count_dark_in_mouth(img: np.ndarray) -> int:
    """Count dark (mouth inside color) pixels inside the mouth ROI."""
    h, w = img.shape[:2]
    cx, cy = w // 2, int(h * 0.7125)
    roi = img[max(0, cy - 20) : min(h, cy + 20), max(0, cx - 60) : min(w, cx + 60)]
    return int(np.sum(np.all(roi < 100, axis=2)))


def _brow_centroid_y(img: np.ndarray, side: str = 'left') -> float:
    """Y centroid of the dark eyebrow pixels.

    Eyebrow is at y = 210 - 62 - 22*b: 148 (b=0) to 126 (b=1), x = 145 (left).
    """
    h, w = img.shape[:2]
    cx = w // 2
    if side == 'left':
        x0, x1 = cx - 90, cx - 10
    else:
        x0, x1 = cx + 10, cx + 90
    y0, y1 = 110, 170
    roi = img[y0:y1, x0:x1]
    mask = np.all(roi < 100, axis=2)
    if not np.any(mask):
        return float('nan')
    ys, xs = np.nonzero(mask)
    return float(y0 + ys.mean())


def _eye_centroids(img: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Centroids of the white eye ellipses (left, right), searched wide."""
    h, w = img.shape[:2]
    cx, cy = w // 2, int(h * 0.525)
    out: list[tuple[float, float]] = []
    for x0, x1 in ((cx - 100, cx), (cx, cx + 100)):
        y0, y1 = cy - 60, cy + 30
        roi = img[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
        mask = np.all(roi > 200, axis=2)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            out.append((float('nan'), float('nan')))
        else:
            out.append((float(x0 + xs.mean()), float(y0 + ys.mean())))
    return (out[0], out[1])


def _mouth_row(img: np.ndarray) -> float:
    """Row with most dark mouth pixels, searched around the expected mouth ROI."""
    h, w = img.shape[:2]
    cx = w // 2
    # Mouth is at C + (0, 85) -> y ~ 295, x ~ 158-242. Avoid head outline (y>340).
    y0, y1 = 270, 320
    x0, x1 = cx - 50, cx + 50
    roi = img[y0:y1, x0:x1]
    dark = np.sum(np.all(roi < 100, axis=2), axis=1)
    if dark.max() == 0:
        return float((y0 + y1) / 2)
    return float(y0 + int(np.argmax(dark)))


# -- shape / dtype -------------------------------------------------------


def test_shape_and_dtype() -> None:
    img = _img()
    assert img.shape == (400, 400, 3)
    assert img.dtype == np.uint8


# -- eyes ----------------------------------------------------------------


def test_eye_l_one_to_zero_reduces_white_pixels() -> None:
    img_open = _img({'eye_l': 1.0, 'eye_r': 1.0})
    img_closed = _img({'eye_l': 0.0, 'eye_r': 1.0})
    w_open = _count_white_in_eye(img_open, 'left')
    w_closed = _count_white_in_eye(img_closed, 'left')
    assert w_open > 0
    assert w_closed < 0.10 * w_open, (w_open, w_closed)
    # Right eye unchanged
    diff_right = abs(
        _count_white_in_eye(img_open, 'right') - _count_white_in_eye(img_closed, 'right')
    )
    assert diff_right < 5


def test_blink_equals_eye_zero() -> None:
    img_blink = _img({'blink': 1, 'eye_l': 1.0, 'eye_r': 1.0})
    img_open = _img({'eye_l': 1.0, 'eye_r': 1.0})
    w_open_l = _count_white_in_eye(img_open, 'left')
    w_open_r = _count_white_in_eye(img_open, 'right')
    assert _count_white_in_eye(img_blink, 'left') < 0.10 * w_open_l
    assert _count_white_in_eye(img_blink, 'right') < 0.10 * w_open_r


# -- mouth ---------------------------------------------------------------


def test_mouth_open_increases_dark_pixels() -> None:
    img_closed = _img({'mouth_open': 0.0})
    img_open = _img({'mouth_open': 1.0})
    d_closed = _count_dark_in_mouth(img_closed)
    d_open = _count_dark_in_mouth(img_open)
    assert d_open >= 5 * d_closed, (d_closed, d_open)


# -- eyebrows ------------------------------------------------------------


def test_brow_l_zero_to_one_raises_centroid() -> None:
    img_b0 = _img({'brow_l': 0.0})
    img_b1 = _img({'brow_l': 1.0})
    y0 = _brow_centroid_y(img_b0, 'left')
    y1 = _brow_centroid_y(img_b1, 'left')
    assert not np.isnan(y0) and not np.isnan(y1)
    assert y0 - y1 >= 15.0, (y0, y1)


# -- head pose -----------------------------------------------------------


def test_head_roll_rotates_eye_centroids() -> None:
    img_0 = _img({'head_roll': 0.0, 'eye_l': 1.0, 'eye_r': 1.0})
    img_20 = _img({'head_roll': 20.0, 'eye_l': 1.0, 'eye_r': 1.0})
    (x0_0, y0_0), (x1_0, y1_0) = _eye_centroids(img_0)
    (x0_20, y0_20), (x1_20, y1_20) = _eye_centroids(img_20)
    cx, cy = 200.0, 210.0
    # Rotate (x0_0, y0_0) by +20 deg around C and compare
    a = math.radians(20.0)
    x_exp = cx + (x0_0 - cx) * math.cos(a) - (y0_0 - cy) * math.sin(a)
    y_exp = cy + (x0_0 - cx) * math.sin(a) + (y0_0 - cy) * math.cos(a)
    assert abs(x0_20 - x_exp) < 3.0, (x0_20, x_exp)
    assert abs(y0_20 - y_exp) < 3.0, (y0_20, y_exp)


def test_yaw_shifts_eye_centroids() -> None:
    img_0 = _img({'head_yaw': 0.0, 'eye_l': 1.0, 'eye_r': 1.0})
    img_30 = _img({'head_yaw': 30.0, 'eye_l': 1.0, 'eye_r': 1.0})
    (x0_0, _), (x1_0, _) = _eye_centroids(img_0)
    (x0_30, _), (x1_30, _) = _eye_centroids(img_30)
    expected_shift = 38.0 * 30.0 / 45.0
    assert abs((x0_30 - x0_0) - expected_shift) < 4.0, (x0_30, x0_0, expected_shift)
    assert abs((x1_30 - x1_0) - expected_shift) < 4.0, (x1_30, x1_0, expected_shift)


def test_pitch_lowers_mouth_centroid() -> None:
    img_0 = _img({'head_pitch': 0.0, 'mouth_open': 0.5})
    img_30 = _img({'head_pitch': 30.0, 'mouth_open': 0.5})
    r0 = _mouth_row(img_0)
    r30 = _mouth_row(img_30)
    expected = 38.0 * 30.0 / 45.0
    assert abs((r30 - r0) - expected) < 4.0, (r0, r30, expected)


# -- missing parameters --------------------------------------------------


def test_missing_params_do_not_error() -> None:
    img = _img(None)
    assert img.shape == (400, 400, 3)
    img2 = _img({})
    assert img2.shape == (400, 400, 3)
