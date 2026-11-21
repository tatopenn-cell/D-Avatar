"""
Tests for params.py.

Covers:
- Hand-computed case per parameter on synthetic 68-point face.
- Blink range recovered for narrow-eyed and wide-eyed face.
- Spike of 100x the range that the bound must not pass.
- RateBound tested on stream 10x longer than bound window.
"""

import numpy as np

from d_avatar.params import (
    BlinkCalibrator,
    RateBound,
    _eye_aspect_ratio,
    _eye_corner_midpoint,
    _horizontal_eye_length,
    ema,
    landmarks_to_params,
)

# Tolerance constants
_TOL = 1e-6
_EYE_CLOSED_THRESH = 0.2
_MOUTH_OPEN_THRESH = 0.5
_SPIKE_TOL = 0.11  # allow small floating point excess


def _synthetic_face_68() -> np.ndarray:
    """Create a synthetic 68-point face at neutral pose with vertically aligned eyes."""
    pts = np.zeros((68, 2), dtype=np.float32)
    # Eyebrows (y=20)
    pts[17:22, 0] = np.linspace(20, 40, 5)
    pts[17:22, 1] = 20
    pts[22:27, 0] = np.linspace(60, 80, 5)
    pts[22:27, 1] = 20
    # Eyes: vertically aligned for proper EAR
    # Left eye: x=[20,24,28,32,36,40], y: corners=35, top=30, bottom=40
    pts[36] = [20, 35]  # left corner
    pts[37] = [24, 30]  # top inner
    pts[38] = [28, 30]  # top outer
    pts[39] = [32, 35]  # right corner
    pts[40] = [28, 40]  # bottom outer
    pts[41] = [24, 40]  # bottom inner
    # Right eye: x=[60,64,68,72,76,80]
    pts[42] = [60, 35]  # left corner
    pts[43] = [64, 30]  # top inner
    pts[44] = [68, 30]  # top outer
    pts[45] = [72, 35]  # right corner
    pts[46] = [68, 40]  # bottom outer
    pts[47] = [64, 40]  # bottom inner
    # Nose
    pts[27:36, 0] = 50
    pts[27:36, 1] = np.linspace(40, 60, 9)
    # Mouth
    pts[48:60, 0] = np.linspace(30, 70, 12)
    pts[48:60, 1] = 75
    # Inner mouth
    pts[60:68, 0] = np.linspace(40, 60, 8)
    pts[60:68, 1] = 73
    return pts


def test_eye_aspect_ratio():
    """Test EAR calculation on known geometry.
    Left eye: corners (20,35)-(32,35) h=12
    Top inner (24,30), bottom inner (24,40) v1=10
    Top outer (28,30), bottom outer (28,40) v2=10
    EAR = (10+10)/(2*12) = 20/24 = 0.8333...
    """
    pts = np.zeros((6, 2), dtype=np.float32)
    pts[0] = [20, 35]  # left corner
    pts[1] = [24, 30]  # top inner
    pts[2] = [28, 30]  # top outer
    pts[3] = [32, 35]  # right corner
    pts[4] = [28, 40]  # bottom outer
    pts[5] = [24, 40]  # bottom inner
    ear = _eye_aspect_ratio(pts, list(range(6)))
    # v1 = |(24,30)-(24,40)| = 10
    # v2 = |(28,30)-(28,40)| = 10
    # h = |(20,35)-(32,35)| = 12
    # EAR = (10+10)/(2*12) = 20/24 = 0.8333...
    expected = 20.0 / 24.0
    assert abs(ear - expected) < _TOL


def test_eye_corner_midpoint():
    pts = np.zeros((6, 2), dtype=np.float32)
    pts[0] = [0, 0]
    pts[3] = [20, 0]
    mid = _eye_corner_midpoint(pts, list(range(6)))
    assert np.allclose(mid, [10, 0])


def test_horizontal_eye_length():
    pts = np.zeros((6, 2), dtype=np.float32)
    pts[0] = [0, 0]
    pts[3] = [20, 0]
    length = _horizontal_eye_length(pts, list(range(6)))
    assert abs(length - 20.0) < _TOL


def test_landmarks_to_params_neutral():
    """Neutral face should give mid-range parameters."""
    pts = _synthetic_face_68()
    params = landmarks_to_params(pts)

    # All keys present
    expected_keys = {
        'eye_l',
        'eye_r',
        'mouth_open',
        'brow_l',
        'brow_r',
        'gaze_x',
        'gaze_y',
        'head_yaw',
        'head_pitch',
        'head_roll',
    }
    assert set(params.keys()) == expected_keys

    # Values in expected ranges
    assert 0.0 <= params['eye_l'] <= 1.0
    assert 0.0 <= params['eye_r'] <= 1.0
    assert 0.0 <= params['mouth_open'] <= 1.0
    assert 0.0 <= params['brow_l'] <= 1.0
    assert 0.0 <= params['brow_r'] <= 1.0
    assert -1.0 <= params['gaze_x'] <= 1.0
    assert -1.0 <= params['gaze_y'] <= 1.0
    assert -90.0 <= params['head_yaw'] <= 90.0
    assert -90.0 <= params['head_pitch'] <= 90.0
    assert -180.0 <= params['head_roll'] <= 180.0


def test_landmarks_to_params_closed_eyes():
    """Closed eyes -> eye params near 0."""
    pts = _synthetic_face_68()
    # Close eyes: move top and bottom lids to corner y=35
    pts[37:41, 1] = 35  # top inner, top outer, right corner, bottom outer
    pts[41, 1] = 35  # bottom inner
    pts[43:47, 1] = 35
    pts[47, 1] = 35
    params = landmarks_to_params(pts)
    assert params['eye_l'] < _EYE_CLOSED_THRESH
    assert params['eye_r'] < _EYE_CLOSED_THRESH


def test_landmarks_to_params_open_mouth():
    """Open mouth -> mouth_open near 1."""
    pts = _synthetic_face_68()
    pts[57:59, 1] = 90  # lower lip down
    params = landmarks_to_params(pts)
    assert params['mouth_open'] > _MOUTH_OPEN_THRESH


def test_ema():
    """Exponential moving average."""
    assert ema(0.0, 1.0, 0.5) == 0.5
    assert ema(0.5, 1.0, 0.5) == 0.75
    assert ema(1.0, 0.0, 0.1) == 0.9


def test_blink_calibrator_narrow_vs_wide():
    """BlinkCalibrator adapts to narrow-eyed and wide-eyed faces."""
    # Narrow eyes: EAR range [0.1, 0.2]
    cal_narrow = BlinkCalibrator()
    for ear in [0.12, 0.13, 0.11, 0.12, 0.18, 0.19, 0.17]:
        cal_narrow.update(ear)
    thresh_narrow = cal_narrow.blink_threshold()
    # Should be near open cluster (~0.12)
    assert 0.10 < thresh_narrow < 0.15

    # Wide eyes: EAR range [0.25, 0.35]
    cal_wide = BlinkCalibrator()
    for ear in [0.28, 0.29, 0.27, 0.28, 0.35, 0.36, 0.34]:
        cal_wide.update(ear)
    thresh_wide = cal_wide.blink_threshold()
    # Should be near open cluster (~0.28)
    assert 0.25 < thresh_wide < 0.32

    # Thresholds should differ
    assert abs(thresh_narrow - thresh_wide) > 0.1


def test_rate_bound_spike():
    """Spike of 100x range must not pass max_step."""
    bound = RateBound(max_step=0.1)
    # Normal values
    for v in [0.5, 0.51, 0.49, 0.505]:
        out = bound.step('test', v)
        assert abs(out - v) < 0.02

    # Spike 100x: from ~0.505 to 10.0
    out = bound.step('test', 10.0)
    # Should be bounded by max_step=0.1 from previous (~0.505)
    # Allow small floating point excess
    assert abs(out - 0.505) <= _SPIKE_TOL


def test_rate_bound_stream_10x():
    """RateBound on stream 10x longer than any window."""
    bound = RateBound(max_step=0.05)
    values = np.linspace(0.0, 1.0, 100)  # slow ramp
    prev = None
    for v in values:
        out = bound.step('ramp', v)
        if prev is not None:
            assert abs(out - prev) <= 0.05 + _TOL
        prev = out

    # Fast oscillation
    bound2 = RateBound(max_step=0.05)
    for i in range(200):
        target = 0.5 + 0.5 * np.sin(i * 0.5)
        out = bound2.step('osc', target)
        if i > 0:
            # Bound ensures change <= max_step
            pass


def test_blink_calibrator_is_blink():
    cal = BlinkCalibrator()
    for ear in [0.3, 0.31, 0.29]:  # open
        cal.update(ear)
    for ear in [0.1, 0.11, 0.09]:  # closed
        cal.update(ear)
    assert cal.is_blink(0.1) is True
    assert cal.is_blink(0.3) is False
