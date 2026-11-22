"""Tests for params.py, with expected values written out from the formulas."""

import numpy as np
import pytest

from d_avatar.params import (
    BlinkCalibrator,
    RateBound,
    ema,
    eye_aspect_ratio,
    landmarks_to_params,
)
from d_avatar.synthetic import synthetic_face_68

_TOL = 1e-6


def test_eye_aspect_ratio_formula():
    pts = np.array([[20, 35], [23, 33.5], [27, 33.5], [30, 35], [27, 36.5], [23, 36.5]], float)
    assert abs(eye_aspect_ratio(pts, (0, 1, 2, 3, 4, 5)) - (3 + 3) / (2 * 10)) < _TOL


def test_neutral_face_parameters():
    p = landmarks_to_params(synthetic_face_68(), ear_open=0.3)
    assert abs(p['eye_l'] - 0.3 / 0.3) < _TOL
    assert abs(p['eye_r'] - 1.0) < _TOL
    # Synthetic face uses old formula with explicit middle points: mouth_open = 0.3
    assert abs(p['mouth_open'] - 0.3) < _TOL
    assert abs(p['brow_l'] - ((15 / 10) - 1.0) / (2.0 - 1.0)) < _TOL
    assert abs(p['brow_r'] - 0.5) < _TOL
    assert abs(p['head_yaw']) < _TOL
    assert abs(p['head_pitch'] - ((40.625 - 35.0) / 40.0) * 60.0) < _TOL
    assert abs(p['head_roll']) < _TOL
    assert 'gaze_x' not in p


def test_closed_eyes_are_zero():
    pts = synthetic_face_68()
    pts[37:42, 1] = 35.0
    p = landmarks_to_params(pts, ear_open=0.3)
    assert p['eye_l'] == 0.0


def test_head_roll_of_rotated_face():
    pts = synthetic_face_68()
    c = pts.mean(axis=0)
    a = np.radians(10.0)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = landmarks_to_params((pts - c) @ r.T + c)
    assert abs(p['head_roll'] - 10.0) < _TOL


def test_shape_error():
    with pytest.raises(ValueError):
        landmarks_to_params(np.zeros((5, 2)))


def test_ema():
    assert ema(0.0, 1.0, 0.5) == 0.5
    assert ema(0.5, 1.0, 0.5) == 0.75
    assert ema(1.0, 0.0, 0.1) == 0.9


@pytest.mark.parametrize('open_ear, closed_ear', [(0.18, 0.05), (0.35, 0.10)])
def test_blink_calibrator_same_code_narrow_and_wide(open_ear, closed_ear):
    cal = BlinkCalibrator()
    rng = np.random.default_rng(0)
    stream = np.where(rng.random(200) < 0.1, closed_ear, open_ear) * (
        1 + 0.02 * rng.standard_normal(200)
    )
    stream[0] = open_ear
    for ear in stream:
        cal.update(float(ear))
    scale = cal.open_ear()
    assert min(open_ear / scale, 1.0) > 0.95
    assert closed_ear / scale < 0.4
    assert cal.is_blink(closed_ear)
    assert not cal.is_blink(open_ear)


def test_rate_bound_spike_and_first_sample():
    bound = RateBound(max_step=0.1)
    assert bound.step('a', 0.5) == 0.5
    assert abs(bound.step('a', 10.0) - 0.6) < _TOL
    assert abs(bound.step('a', -10.0) - 0.5) < _TOL


def test_rate_bound_per_name_and_ramp_ten_times_longer():
    bound = RateBound(max_step=0.05, per_name={'head_yaw': 5.0})
    prev = {'x': None, 'head_yaw': None}
    for i in range(500):
        for name, target in (('x', 1.0), ('head_yaw', 30.0)):
            out = bound.step(name, target if i else 0.0)
            if prev[name] is not None:
                assert abs(out - prev[name]) <= (0.05 if name == 'x' else 5.0) + _TOL
            prev[name] = out
    assert abs(prev['x'] - 1.0) < _TOL
    assert abs(prev['head_yaw'] - 30.0) < _TOL


def test_blink_calibrator_before_first_sample():
    cal = BlinkCalibrator()
    assert cal.open_ear() == 0.3
    assert not cal.is_blink(0.0)
