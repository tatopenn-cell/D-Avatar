"""Tests for the native tracker: synthetic frames, no camera, no ML models.

Every parameter has:
- an assertion against the synthetic truth with tolerance;
- a test that fails if the primitive behind it is replaced by a constant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from d_avatar.tracker import NativeTracker


def _make_frame(
    eyes_open: bool = True,
    mouth_open: bool = False,
    head_dx: float = 0.0,
    head_dy: float = 0.0,
) -> np.ndarray:
    """Draw a synthetic face: white head ellipse, eye ellipses, mouth strip.

    Eyes closed = entire eye ROI filled black.
    Mouth open = dark strip inside the mouth ROI (dark interior = open).
    Coordinates are chosen to land inside the tracker's ROIs for face box
    (40, 10, 120, 110): eyes at y=40..54, mouth at y=80..88.
    """
    img = np.zeros((120, 160), dtype=np.float32)
    cx, cy = int(80 + head_dx), int(60 + head_dy)
    yy, xx = np.ogrid[:120, :160]
    head = ((xx - cx) / 40.0) ** 2 + ((yy - cy) / 50.0) ** 2 <= 1.0
    img[head] = 1.0

    # Eyes: open = white ellipses; closed = whole tracker eye ROI black.
    # Tracker eye ROIs: left=(56,40,64,54), right=(88,40,96,54) for face box
    # (40, 10, 120, 110).
    l_roi = (56, 40, 64, 54)
    r_roi = (88, 40, 96, 54)
    for x0, y0, x1, y1 in (l_roi, r_roi):
        if eyes_open:
            cx_e = x0 + (x1 - x0) / 2
            cy_e = y0 + (y1 - y0) / 2
            eye = ((xx - cx_e) / 4.0) ** 2 + ((yy - cy_e) / 3.0) ** 2 <= 1.0
            img[eye] = 1.0
        else:
            img[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 0.0

    # Mouth: closed = white strip; open = dark strip (dark interior = open)
    if mouth_open:
        img[max(0, cy + 20) : max(0, cy + 26), max(0, cx - 12) : max(0, cx + 12)] = 0.2
    else:
        img[max(0, cy + 20) : max(0, cy + 26), max(0, cx - 12) : max(0, cx + 12)] = 1.0

    return img


def _tracker_with_box() -> NativeTracker:
    t = NativeTracker(max_step=1.0, max_step_deg=90.0, calibrate_frames=0)
    # Fix the face ROI manually: head is centred at (80, 60), 80x100.
    # Tracker eye ROI: y = 10 + int(100*0.30)=40 .. 40+int(100*0.14)=54
    # Tracker mouth ROI: y = 10 + int(100*0.70)=80 .. 80+int(100*0.08)=88
    t.set_face_box((40, 10, 120, 110))
    return t


def test_eyes_open_are_one():
    t = _tracker_with_box()
    p = t(_make_frame(eyes_open=True, mouth_open=False))
    assert abs(p['eye_l'] - 1.0) < 0.15, p
    assert abs(p['eye_r'] - 1.0) < 0.15, p


def test_eyes_closed_are_zero():
    t = _tracker_with_box()
    p = t(_make_frame(eyes_open=False, mouth_open=False))
    assert p['eye_l'] < 0.3, p
    assert p['eye_r'] < 0.3, p


def test_mouth_open_increases():
    t = _tracker_with_box()
    p_closed = t(_make_frame(eyes_open=True, mouth_open=False))
    p_open = t(_make_frame(eyes_open=True, mouth_open=True))
    assert p_open['mouth_open'] > p_closed['mouth_open'] + 0.3, (p_closed, p_open)


def test_blink_on_energy_spike():
    t = _tracker_with_box()
    base = _make_frame(eyes_open=True, mouth_open=False)
    spike = base.copy()
    # Make both tracker eye ROIs fully black (max change from white)
    spike[40:54, 56:64] = 0.0  # left eye ROI black
    spike[40:54, 88:96] = 0.0  # right eye ROI black
    t(base)  # warm up MotionEnergy (needs a previous frame)
    p = t(spike)
    assert p['blink'] == 1.0, p


def test_head_angles_zero_on_still_head():
    t = _tracker_with_box()
    p = t(_make_frame(eyes_open=True, mouth_open=False, head_dx=0.0, head_dy=0.0))
    assert abs(p['head_yaw']) < 5.0, p
    assert abs(p['head_pitch']) < 5.0, p
    assert abs(p['head_roll']) < 5.0, p


def test_returns_all_seven_keys():
    t = _tracker_with_box()
    p = t(_make_frame())
    for k in ('head_yaw', 'head_pitch', 'head_roll', 'mouth_open', 'eye_l', 'eye_r', 'blink'):
        assert k in p, p
    assert p['blink'] in (0.0, 1.0), p


# -- primitive-removed tests ------------------------------------------------


def _constant_features(*args: Any, **kwargs: Any) -> dict[str, float]:
    return {'mean': 0.5, 'contrast': 0.0, 'centroid_x': 0.5, 'centroid_y': 0.5}


def test_eye_l_fails_when_features_is_constant(monkeypatch: pytest.MonkeyPatch):
    t = _tracker_with_box()
    monkeypatch.setattr(t.features, 'transform_one', _constant_features)
    p = t(_make_frame(eyes_open=True, mouth_open=False))
    # With constant features, eye_l collapses to ~0.5, not ~1.0
    assert abs(p['eye_l'] - 1.0) > 0.3, p


def test_mouth_fails_when_features_is_constant(monkeypatch: pytest.MonkeyPatch):
    t = _tracker_with_box()
    monkeypatch.setattr(t.features, 'transform_one', _constant_features)
    p_closed = t(_make_frame(eyes_open=True, mouth_open=False))
    p_open = t(_make_frame(eyes_open=True, mouth_open=True))
    # With constant features, open and closed mouth are identical
    assert abs(p_open['mouth_open'] - p_closed['mouth_open']) < 0.05, (p_closed, p_open)


def test_blink_fails_when_motion_energy_is_constant(monkeypatch: pytest.MonkeyPatch):
    t = _tracker_with_box()
    monkeypatch.setattr(t.me_l, 'transform_one', lambda *a, **k: {'motion_max': 0.0})
    monkeypatch.setattr(t.me_r, 'transform_one', lambda *a, **k: {'motion_max': 0.0})
    base = _make_frame(eyes_open=True, mouth_open=False)
    spike = base.copy()
    spike[40:54, 56:64] = 0.0  # left eye ROI black
    spike[40:54, 88:96] = 0.0  # right eye ROI black
    t(base)
    p = t(spike)
    assert p['blink'] == 0.0, p
