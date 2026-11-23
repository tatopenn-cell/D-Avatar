"""2D avatar drawer: pure function from parameters to a BGR image.

Pure function: reads only ``params``, returns a fresh BGR uint8 image of
shape ``(size, size, 3)``. Missing parameters are neutral.

No ML, no filtering: every parameter maps directly to geometry.
Optional extras beyond the base 7 parameters:
- ``smile`` (0, 1): corners of the mouth pulled up (0 = neutral, 1 = smile)
- ``time`` (float): seconds, used for automatic blinking when ``blink`` is
  absent or 0
- ``style`` (dict): custom BGR colors. Keys: ``skin_color``, ``skin_dark``,
  ``skin_light``, ``hair_color``, ``hair_light``, ``brow_color``,
  ``mouth_inside``, ``mouth_outline``, ``lip_color``, ``eye_color``,
  ``pupil_color``, ``eye_reflect``, ``eye_closed``.

Uses OpenCV only inside this module (optional extra, per SKILL.md).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import cv2  # noqa: PLC0415

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - cv2 is optional for tests without drawing
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

_CENTER = (200, 210)
_SKIN = (195, 175, 155)  # BGR
_SKIN_DARK = (150, 130, 110)  # BGR
_SKIN_LIGHT = (225, 210, 195)  # BGR
_BG = (235, 235, 235)
_EYE_CLOSED_COLOR = (30, 30, 30)
_IRIS_COLOR = (70, 70, 70)
_PUPIL_COLOR = (10, 10, 10)
_EYE_REFLECT_COLOR = (240, 240, 240)
_BROW_COLOR = (60, 50, 40)
_MOUTH_INSIDE = (50, 35, 80)
_MOUTH_OUTLINE = (0, 0, 140)
_LIP_COLOR = (180, 90, 100)
_NECK_COLOR = (205, 205, 205)
_HAIR_COLOR = (25, 20, 15)
_HAIR_LIGHT = (55, 45, 35)
_SHADOW_COLOR = (180, 175, 170)

# Blink automation: eyes close for ~0.15 s every 3-6 s
_BLINK_INTERVAL_MIN = 3.0
_BLINK_INTERVAL_MAX = 6.0
_BLINK_DURATION = 0.15
_BROW_LOW = 0.4
_BROW_HIGH = 0.7
_MOUTH_CLOSED_THRESHOLD = 0.05
_SMILE_CLOSED_THRESHOLD = 0.1

# Deterministic blink state shared across calls (no camera, no ML needed).
_blink_last_times: list[float] = []


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


def _auto_blink(time_s: float, last_blink: list[float]) -> float:
    """Return 1.0 if eyes should be closed right now (automatic blink)."""
    if not last_blink:
        last_blink.append(time_s)
        return 0.0
    # Pick a deterministic next blink time from the time base.
    cycle = (time_s * 7.3) % (_BLINK_INTERVAL_MAX + _BLINK_INTERVAL_MIN)
    next_blink = last_blink[-1] + _BLINK_INTERVAL_MIN + cycle * 0.6
    if time_s >= next_blink:
        # Blink now: closed for _BLINK_DURATION seconds.
        if time_s - next_blink < _BLINK_DURATION:
            return 1.0
        last_blink.append(next_blink)
    return 0.0


def draw_avatar(  # noqa: PLR0915
    params: dict[str, Any] | None,
    size: int = 400,
    style: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw a 2D avatar from parameters.

    Pure function: reads only ``params``, returns a fresh BGR uint8 image of
    shape ``(size, size, 3)``. Missing parameters are neutral.

    Args:
        params: dict with keys head_yaw, head_pitch, head_roll (deg),
            mouth_open, eye_l, eye_r (0, 1), blink (0 or 1), brow_l, brow_r
            (0, 1), gaze_x, gaze_y (-1, 1), smile (0, 1), time (seconds).
            Missing keys use neutral values.
        size: output image size in pixels (default 400).
        style: optional dict with custom colors. Keys:
            ``skin_color``, ``skin_dark``, ``skin_light``,
            ``hair_color``, ``hair_light``, ``brow_color``,
            ``mouth_inside``, ``mouth_outline``, ``lip_color``,
            ``eye_color``, ``pupil_color``, ``eye_reflect``,
            ``eye_closed``. Values are BGR tuples or lists.

    Returns:
        BGR uint8 array of shape (size, size, 3).
    """
    if _CV2_AVAILABLE is False:
        raise ImportError(
            'opencv-python is required for draw_avatar. Install with: pip install opencv-python'
        )

    p = params or {}
    style = style or {}

    def _col(key: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        v = style.get(key, default)
        return (int(v[0]), int(v[1]), int(v[2]))

    skin = _col('skin_color', _SKIN)
    skin_dark = _col('skin_dark', _SKIN_DARK)
    skin_light = _col('skin_light', _SKIN_LIGHT)
    hair_color = _col('hair_color', _HAIR_COLOR)
    hair_light = _col('hair_light', _HAIR_LIGHT)
    brow_color = _col('brow_color', _BROW_COLOR)
    mouth_inside = _col('mouth_inside', _MOUTH_INSIDE)
    mouth_outline = _col('mouth_outline', _MOUTH_OUTLINE)
    lip_color = _col('lip_color', _LIP_COLOR)
    iris_color = _col('eye_color', _IRIS_COLOR)
    pupil_color = _col('pupil_color', _PUPIL_COLOR)
    eye_reflect = _col('eye_reflect', _EYE_REFLECT_COLOR)
    eye_closed = _col('eye_closed', _EYE_CLOSED_COLOR)

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
    smile = _get(p, 'smile', 0.0)
    time_s = _get(p, 'time', 0.0)

    # Automatic blink when blink is neutral (0) and no explicit blink requested.
    if blink == 0.0 and time_s > 0:
        blink = _auto_blink(time_s, _blink_last_times)

    if blink == 1:
        eye_l = 0.0
        eye_r = 0.0

    s = size / 400.0
    C = _point(_CENTER, size)

    img: np.ndarray = np.full((size, size, 3), _BG, dtype=np.uint8)

    # ---------------------------------------------------------------- shadow
    shadow_ell = (
        _point((C[0], C[1] + 165), size),
        (int(100 * s), int(18 * s)),
        head_roll,
    )
    cv2.ellipse(
        img,
        shadow_ell[0],
        shadow_ell[1],
        shadow_ell[2],
        0,
        360,
        _SHADOW_COLOR,
        -1,
    )

    # ---------------------------------------------------------------- neck
    neck_w = int(_scale(70, size))
    neck_h = int(_scale(60, size))
    neck_x = C[0] - neck_w // 2
    neck_y = C[1] + int(_scale(150, size))
    cv2.rectangle(
        img,
        (neck_x, neck_y),
        (neck_x + neck_w, neck_y + neck_h),
        _NECK_COLOR,
        -1,
    )
    cv2.rectangle(
        img,
        (neck_x, neck_y),
        (neck_x + neck_w, neck_y + neck_h),
        skin_dark,
        1,
    )

    # ---------------------------------------------------------------- head
    ax = int(_scale(120, size))
    ay = int(_scale(150, size))
    cv2.ellipse(img, C, (ax, ay), head_roll, 0, 360, skin, -1)
    cv2.ellipse(img, C, (ax, ay), head_roll, 0, 360, (0, 0, 0), 2)

    # ---------------------------------------------------------------- hair
    hair_w = int(_scale(110, size))
    hair_h = int(_scale(95, size))
    hair_c = _point((C[0], C[1] - 70), size)
    # Main hair mass (covers top of head).
    cv2.ellipse(
        img,
        (hair_c[0], hair_c[1] + int(_scale(10, size))),
        (hair_w, hair_h),
        head_roll,
        180,
        360,
        hair_color,
        -1,
    )
    # Hair bangs / fringe across forehead.
    for i in range(-2, 3):
        bx = C[0] + int(i * 16 * s)
        by = C[1] - int(_scale(55, size))
        cv2.ellipse(
            img,
            (bx, by),
            (int(_scale(18, size)), int(_scale(22, size))),
            head_roll + i * 8,
            180,
            360,
            hair_color,
            -1,
        )
    # Hair light highlight (left side).
    hl_c = _point((C[0] - 30, C[1] - 75), size)
    cv2.ellipse(
        img,
        hl_c,
        (int(_scale(30, size)), int(_scale(40, size))),
        head_roll,
        0,
        180,
        hair_light,
        -1,
    )

    # Per-feature offsets from yaw/pitch.
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

    # ---------------------------------------------------------------- ears
    ear_ax = int(_scale(14, size))
    ear_ay = int(_scale(28, size))
    for ex in (_CENTER[0] - 122, _CENTER[0] + 122):
        ear_c = _point((ex, _CENTER[1]), size)
        cv2.ellipse(img, ear_c, (ear_ax, ear_ay), head_roll, 0, 360, skin, -1)
        cv2.ellipse(img, ear_c, (ear_ax, ear_ay), head_roll, 0, 360, (0, 0, 0), 1)

    # ---------------------------------------------------------------- eyes
    for _name, e, base_x in (
        ('eye_l', eye_l, _CENTER[0] - 55),
        ('eye_r', eye_r, _CENTER[0] + 55),
    ):
        eye_open = max(0.06, e) if e > 0 else 0.0
        ec = feat_center((base_x, _CENTER[1] - 20))
        eax, eay = feat_axes(30.0, 22.0 * eye_open)
        if blink == 1 or e == 0:
            # Closed: dark line with slight curve.
            cv2.line(
                img,
                (ec[0] - eax, ec[1]),
                (ec[0] + eax, ec[1]),
                eye_closed,
                3,
            )
        else:
            # Sclera.
            cv2.ellipse(img, ec, (eax, eay), head_roll, 0, 360, (255, 255, 255), -1)
            # Iris.
            iris_c = (
                ec[0] + int(gaze_x * 12 * s),
                ec[1] + int(gaze_y * 6 * s),
            )
            cv2.circle(img, iris_c, 10, iris_color, -1)
            cv2.circle(img, iris_c, 4, pupil_color, -1)
            # Eye reflection (catchlight).
            refl = (iris_c[0] - 3, iris_c[1] - 3)
            cv2.circle(img, refl, 3, eye_reflect, -1)
            # Upper lid shadow line.
            lid_top = (ec[0] - eax, ec[1] - eay + 2)
            lid_end = (ec[0] + eax, ec[1] - eay + 2)
            cv2.line(img, lid_top, lid_end, eye_closed, 1)

    # ---------------------------------------------------------------- eyebrows
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
        cv2.line(img, p1, p2, brow_color, th)
        # Brow highlight (subtle).
        cv2.line(img, p1, p2, skin_light, 1)

    # ---------------------------------------------------------------- nose
    nose_c = feat_center((_CENTER[0], _CENTER[1] - 5))
    pts = np.array(
        [
            nose_c,
            feat_center((_CENTER[0] - 14, _CENTER[1] + 35)),
            feat_center((_CENTER[0] + 14, _CENTER[1] + 35)),
        ]
    )
    cv2.fillConvexPoly(img, pts, skin_dark)
    cv2.polylines(img, [pts], True, skin_dark, 1)
    # Nostrils.
    nl = feat_center((_CENTER[0] - 8, _CENTER[1] + 28))
    nr = feat_center((_CENTER[0] + 8, _CENTER[1] + 28))
    cv2.circle(img, nl, 2, (100, 80, 70), -1)
    cv2.circle(img, nr, 2, (100, 80, 70), -1)

    # ---------------------------------------------------------------- mouth
    mouth_c = feat_center((_CENTER[0], _CENTER[1] + 85))
    m_ax = int(_scale(42, size))
    m_ay = int(_scale(3 + 32 * mouth_open, size))
    # Smile pulls corners up.
    smile_lift = int(smile * 12 * s)
    mc_y = mouth_c[1] - smile_lift
    if mouth_open < _MOUTH_CLOSED_THRESHOLD and smile < _SMILE_CLOSED_THRESHOLD:
        # Closed mouth: thin line, slightly curved by smile.
        cv2.line(
            img,
            (mouth_c[0] - m_ax, mc_y),
            (mouth_c[0] + m_ax, mc_y + int(smile * 6 * s)),
            mouth_outline,
            2,
        )
    else:
        # Open mouth.
        cv2.ellipse(
            img,
            (mouth_c[0], mc_y),
            (m_ax, m_ay),
            head_roll,
            0,
            360,
            mouth_inside,
            -1,
        )
        cv2.ellipse(
            img,
            (mouth_c[0], mc_y),
            (m_ax, m_ay),
            head_roll,
            0,
            360,
            mouth_outline,
            2,
        )
        # Lips: darker outline.
        lip_h = max(1, int(m_ay * 0.35))
        cv2.ellipse(
            img,
            (mouth_c[0], mc_y),
            (m_ax, lip_h),
            head_roll,
            0,
            360,
            lip_color,
            -1,
        )

    return img
