"""2D avatar drawer: pure function from parameters to a BGR image.

Uses OpenCV only inside this module (optional extra, per SKILL.md).
No filtering, no ML: every parameter maps directly to geometry.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - cv2 is optional for tests without drawing
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

_CENTER = (200, 210)
_SKIN = (200, 180, 160)  # BGR
_SKIN_DARK = (160, 140, 120)
_BG = (230, 230, 230)
_EYE_CLOSED_COLOR = (30, 30, 30)
_IRIS_COLOR = (60, 60, 60)
_PUPIL_COLOR = (0, 0, 0)
_BROW_COLOR = (50, 50, 50)
_MOUTH_INSIDE = (40, 30, 70)
_MOUTH_OUTLINE = (0, 0, 128)
_NECK_COLOR = (200, 200, 200)
_BROW_LOW = 0.4
_BROW_HIGH = 0.7
_MOUTH_CLOSED_THRESHOLD = 0.05


def _scale(value: float, size: int) -> float:
    return value * size / 400.0


def _point(p: tuple[float, float], size: int) -> tuple[int, int]:
    return (int(p[0] * size / 400.0), int(p[1] * size / 400.0))


def _rot(
    point: tuple[float, float],
    center: tuple[float, float],
    angle_deg: float,
) -> tuple[float, float]:
    a = math.radians(angle_deg)
    x, y = point[0] - center[0], point[1] - center[1]
    return (
        center[0] + x * math.cos(a) - y * math.sin(a),
        center[1] + x * math.sin(a) + y * math.cos(a),
    )


def _get(params: dict[str, Any], name: str, default: float) -> float:
    v = params.get(name, default)
    return float(v) if v is not None else default


def draw_avatar(  # noqa: PLR0915
    params: dict[str, Any] | None, size: int = 400
) -> np.ndarray:
    """Draw a 2D avatar from parameters.

    Pure function: reads only ``params``, returns a fresh BGR uint8 image of
    shape ``(size, size, 3)``. Missing parameters are neutral.

    Args:
        params: dict with keys head_yaw, head_pitch, head_roll (deg),
            mouth_open, eye_l, eye_r (0, 1), blink (0 or 1), brow_l, brow_r
            (0, 1), gaze_x, gaze_y (-1, 1). Missing keys use neutral values.
        size: output image size in pixels (default 400).

    Returns:
        BGR uint8 array of shape (size, size, 3).
    """
    if _CV2_AVAILABLE is False:
        raise ImportError(
            'opencv-python is required for draw_avatar. Install with: pip install opencv-python'
        )

    p = params or {}

    head_yaw = _get(p, 'head_yaw', 0.0)
    head_pitch = _get(p, 'head_pitch', 0.0)
    head_roll = _get(p, 'head_roll', 0.0)
    mouth_open = _get(p, 'mouth_open', 0.0)
    eye_l = _get(p, 'eye_l', 1.0)
    eye_r = _get(p, 'eye_r', 1.0)
    blink = _get(p, 'blink', 0.0)
    brow_l = _get(p, 'brow_l', 0.5)
    brow_r = _get(p, 'brow_r', 0.5)
    gaze_x = _get(p, 'gaze_x', 0.0)
    gaze_y = _get(p, 'gaze_y', 0.0)

    if blink == 1:
        eye_l = 0.0
        eye_r = 0.0

    s = size / 400.0
    C = _point(_CENTER, size)

    img: np.ndarray = np.full((size, size, 3), _BG, dtype=np.uint8)

    # Neck: rectangle 70 x 60 below the head
    neck_w = int(_scale(70, size))
    neck_h = int(_scale(60, size))
    neck_x = C[0] - neck_w // 2
    neck_y = C[1] + int(_scale(150, size))
    cv2.rectangle(img, (neck_x, neck_y), (neck_x + neck_w, neck_y + neck_h), _NECK_COLOR, -1)

    # Head ellipse (full, not affected by yaw/pitch scale, only by roll)
    ax = int(_scale(120, size))
    ay = int(_scale(150, size))
    cv2.ellipse(img, C, (ax, ay), head_roll, 0, 360, _SKIN, -1)
    cv2.ellipse(img, C, (ax, ay), head_roll, 0, 360, (0, 0, 0), 3)

    # Ears
    ear_ax = int(_scale(14, size))
    ear_ay = int(_scale(28, size))
    for ex in (_CENTER[0] - 122, _CENTER[0] + 122):
        ear_c = _point((ex, _CENTER[1]), size)
        cv2.ellipse(img, ear_c, (ear_ax, ear_ay), head_roll, 0, 360, _SKIN, -1)

    # Per-feature offsets from yaw/pitch
    yaw_shift_px = (head_yaw / 45.0) * 38.0 * s
    pitch_shift_px = (head_pitch / 45.0) * 38.0 * s
    yaw_ax_scale = max(0.62, np.cos(np.radians(head_yaw)))
    pitch_ay_scale = max(0.7, np.cos(np.radians(head_pitch)))

    def feat_center(base: tuple[float, float]) -> tuple[int, int]:
        x = base[0] + yaw_shift_px
        y = base[1] + pitch_shift_px
        return _point(_rot((x, y), _CENTER, head_roll), size)

    def feat_axes(ax0: float, ay0: float) -> tuple[int, int]:
        return (int(ax0 * yaw_ax_scale * s), int(ay0 * pitch_ay_scale * s))

    # Eyes
    for _name, e, base_x in (('eye_l', eye_l, _CENTER[0] - 55), ('eye_r', eye_r, _CENTER[0] + 55)):
        eye_open = max(0.06, e) if e > 0 else 0.0
        ec = feat_center((base_x, _CENTER[1] - 20))
        eax, eay = feat_axes(30.0, 22.0 * eye_open)
        if blink == 1 or e == 0:
            # Closed: dark line
            cv2.line(img, (ec[0] - eax, ec[1]), (ec[0] + eax, ec[1]), _EYE_CLOSED_COLOR, 3)
        else:
            cv2.ellipse(img, ec, (eax, eay), head_roll, 0, 360, (255, 255, 255), -1)
            # Iris
            iris_c = (
                ec[0] + int(gaze_x * 14 * s),
                ec[1] + int(gaze_y * 8 * s),
            )
            # Clip iris to eye ellipse (draw small dark circle inside)
            cv2.circle(img, iris_c, 11, _IRIS_COLOR, -1)
            cv2.circle(img, iris_c, 5, _PUPIL_COLOR, -1)

    # Eyebrows
    for _name, b, base_x in (
        ('brow_l', brow_l, _CENTER[0] - 55),
        ('brow_r', brow_r, _CENTER[0] + 55),
    ):
        brow_c = feat_center((base_x, _CENTER[1] - 62 - 22 * b))
        angle = -10.0 if b < _BROW_LOW else (10.0 if b > _BROW_HIGH else 0.0)
        length = int(_scale(64, size))
        th = int(_scale(6, size))
        dx = int(length / 2 * np.cos(np.radians(angle)))
        dy = int(length / 2 * np.sin(np.radians(angle)))
        p1 = (brow_c[0] - dx, brow_c[1] - dy)
        p2 = (brow_c[0] + dx, brow_c[1] + dy)
        cv2.line(img, p1, p2, _BROW_COLOR, th)

    # Nose
    nose_c = feat_center((_CENTER[0], _CENTER[1] - 5))
    pts = np.array(
        [
            nose_c,
            feat_center((_CENTER[0] - 14, _CENTER[1] + 35)),
            feat_center((_CENTER[0] + 14, _CENTER[1] + 35)),
        ]
    )
    cv2.fillConvexPoly(img, pts, _SKIN_DARK)

    # Mouth
    mouth_c = feat_center((_CENTER[0], _CENTER[1] + 85))
    m_ax = int(_scale(42, size))
    m_ay = int(_scale(3 + 32 * mouth_open, size))
    if mouth_open < _MOUTH_CLOSED_THRESHOLD:
        cv2.line(
            img,
            (mouth_c[0] - m_ax, mouth_c[1]),
            (mouth_c[0] + m_ax, mouth_c[1]),
            _MOUTH_OUTLINE,
            3,
        )
    else:
        cv2.ellipse(img, mouth_c, (m_ax, m_ay), head_roll, 0, 360, _MOUTH_INSIDE, -1)
        cv2.ellipse(img, mouth_c, (m_ax, m_ay), head_roll, 0, 360, _MOUTH_OUTLINE, 3)

    return img
