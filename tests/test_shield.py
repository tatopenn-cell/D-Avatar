import numpy as np
from dense_armor.checks import check_estimator
from dense_armor.core.online.timeseries.kalman import KalmanFilter

from d_avatar.shield import FaceShield, _hampel_value

_TOL_1E_9 = 1e-9
_TOL_1E_6 = 1e-6
_OUTLIER_TOL = 1.0
_CLEANED_BOTH = 2


def _frame(pts, conf, i, fps=30.0):
    return {'points': pts, 'confidence': conf, 't': float(i) / fps}


def test_hampel_value():
    val, is_out = _hampel_value([1.0, 2.0, 3.0], 10.0, 3.0)
    assert is_out
    assert abs(val - 2.0) < _TOL_1E_9
    val, is_out = _hampel_value([1.0, 2.0, 3.0], 2.5, 3.0)
    assert not is_out
    assert abs(val - 2.5) < _TOL_1E_9


def test_kalman_1d_hand_computed():
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    q = np.diag([0.01, 0.1])
    r = np.array([[0.1]])
    kf = KalmanFilter(F=f, H=h, Q=q, R=r, x0=np.zeros(2), P0=np.eye(2))
    kf.learn_one({'pos': 1.0})
    s = 2.0 + 0.01 + 0.1
    assert abs(kf._x[0] - (2.01 / s)) < _TOL_1E_6
    assert abs(kf._x[1] - (1.0 / s)) < _TOL_1E_6


def test_face_shield_clean_stream():
    shield = FaceShield(window=7, n_sigmas=3.0, max_gap=5)
    n = 10
    pts = np.tile(np.arange(n, dtype=np.float32).reshape(n, 1), (1, 2)) * 10.0
    conf = np.ones(n, dtype=np.float32)
    for i in range(20):
        out = shield.transform_one(_frame(pts, conf, i))
        assert np.abs(out['points'] - pts).max() < _TOL_1E_6
        assert out['cleaned'].sum() == 0


def test_face_shield_one_frame_outlier():
    shield = FaceShield(window=7, n_sigmas=3.0, max_gap=5)
    base = np.tile(np.arange(5, dtype=np.float32).reshape(5, 1), (1, 2)) * 10.0
    conf = np.ones(5, dtype=np.float32)
    for i in range(10):
        shield.transform_one(_frame(base, conf, i))
    bad = base.copy()
    bad[2] = [999.0, 999.0]
    out = shield.transform_one(_frame(bad, conf, 10))
    assert np.abs(out['points'][2] - base[2]).max() < _OUTLIER_TOL
    assert out['cleaned'][2] == _CLEANED_BOTH


def test_face_shield_moving_occluder_static_face():
    shield = FaceShield(window=7, n_sigmas=3.0, max_gap=5)
    n = 8
    base = np.zeros((n, 2), dtype=np.float32)
    base[:, 0] = np.arange(n) * 10.0
    conf = np.ones(n, dtype=np.float32)
    for i in range(10):
        shield.transform_one(_frame(base, conf, i))
    max_err = 0.0
    for k in range(20):
        c = conf.copy()
        s = k % (n - 1)
        c[s] = 0.0
        c[s + 1] = 0.0
        out = shield.transform_one(_frame(base, c, 10 + k))
        for p in (s, s + 1):
            max_err = max(max_err, float(np.linalg.norm(out['points'][p] - base[p])))
            assert out['cleaned'][p] == _CLEANED_BOTH
    assert max_err < 1.0


def test_face_shield_constant_velocity_during_gap():
    shield = FaceShield(window=7, n_sigmas=3.0, max_gap=10)
    conf = np.ones(1, dtype=np.float32)
    for i in range(60):
        shield.transform_one(_frame(np.array([[i * 3.0, 0.0]], dtype=np.float32), conf, i))
    off = np.zeros(1, dtype=np.float32)
    for i in range(10):
        out = shield.transform_one(_frame(np.array([[0.0, 0.0]], dtype=np.float32), off, 60 + i))
        assert abs(out['points'][0, 0] - (177.0 + (i + 1) * 3.0)) < 1.0


def test_face_shield_variable_fps():
    shield = FaceShield(window=7, n_sigmas=3.0, max_gap=10)
    conf = np.ones(1, dtype=np.float32)
    for i in range(60):
        pt = np.array([[i * 3.0, 0.0]], dtype=np.float32)
        shield.transform_one(_frame(pt, conf, i, fps=15.0))
    off = np.zeros(1, dtype=np.float32)
    for i in range(5):
        pt = np.array([[0.0, 0.0]], dtype=np.float32)
        out = shield.transform_one(_frame(pt, off, 60 + i, fps=15.0))
        assert abs(out['points'][0, 0] - (177.0 + (i + 1) * 3.0)) < 1.0


def test_face_shield_gap_decays_to_neutral():
    neutral = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    shield = FaceShield(window=7, n_sigmas=3.0, max_gap=3, neutral=neutral)
    conf = np.ones(3, dtype=np.float32)
    moved = neutral + 5.0
    for i in range(10):
        shield.transform_one(_frame(neutral, conf, i))
    for i in range(5):
        shield.transform_one(_frame(moved, conf, 10 + i))
    off = np.zeros(3, dtype=np.float32)
    prev = [None] * 3
    for i in range(15):
        out = shield.transform_one(_frame(moved, off, 15 + i))
        if i > 3:
            for p in range(3):
                d = float(np.linalg.norm(out['points'][p] - neutral[p]))
                assert d < float(np.linalg.norm(out['points'][p] - moved[p]))
                if prev[p] is not None and prev[p] > 1e-3:
                    assert d < prev[p] - _TOL_1E_9
                prev[p] = d


def test_face_shield_neutral_first_frame():
    shield = FaceShield()
    base = np.array([[5.0, 7.0], [10.0, 3.0], [20.0, 9.0]], dtype=np.float32)
    shield.transform_one(_frame(base, np.ones(3, dtype=np.float32), 0))
    assert np.allclose(shield.neutral, base, atol=_TOL_1E_6)


def test_face_shield_neutral_partial_occlusion():
    shield = FaceShield()
    base = np.array([[5.0, 7.0], [10.0, 3.0], [20.0, 9.0]], dtype=np.float32)
    conf = np.ones(3, dtype=np.float32)
    conf[0] = 0.0
    out = shield.transform_one(_frame(base, conf, 0))
    assert np.isnan(shield.neutral[0]).all()
    assert np.allclose(shield.neutral[1:], base[1:], atol=_TOL_1E_6)
    assert np.allclose(out['points'][0], base[0])
    shield.transform_one(_frame(base, np.ones(3, dtype=np.float32), 1))
    assert np.allclose(shield.neutral[0], base[0], atol=_TOL_1E_6)


def test_face_shield_point_occluded_before_first_measurement():
    shield = FaceShield()
    pts = np.array([[50.0, 60.0], [10.0, 3.0]], dtype=np.float32)
    out = shield.transform_one(_frame(pts, np.array([0.0, 1.0], dtype=np.float32), 0))
    assert np.allclose(out['points'][0], [50.0, 60.0])


def test_face_shield_clone_with_neutral_then_long_gap():
    neutral = np.zeros((3, 2))
    shield = FaceShield(neutral=neutral, max_gap=2).clone()
    pts = np.array([[5.0, 7.0], [10.0, 3.0], [20.0, 9.0]], dtype=np.float32)
    for i in range(5):
        shield.transform_one(_frame(pts, np.ones(3, dtype=np.float32), i))
    for i in range(6):
        shield.transform_one(_frame(pts, np.zeros(3, dtype=np.float32), 5 + i))


def test_face_shield_hampel_history_reset_after_gap():
    shield = FaceShield(window=7, n_sigmas=3.0, max_gap=20)
    at = np.array([[10.0, 10.0]], dtype=np.float32)
    far = np.array([[40.0, 10.0]], dtype=np.float32)
    for i in range(10):
        shield.transform_one(_frame(at, np.ones(1, dtype=np.float32), i))
    for i in range(10):
        shield.transform_one(_frame(at, np.zeros(1, dtype=np.float32), 10 + i))
    out = shield.transform_one(_frame(far, np.ones(1, dtype=np.float32), 20))
    assert out['cleaned'][0] == 0


def test_check_estimator_face_shield():
    shield = FaceShield()
    result = check_estimator(shield)
    assert result['failed'] == [], result['failed']
    assert result['not_applicable'] == [], result['not_applicable']
    assert shield._unit_test_skips() == set()


def test_flat_dict_is_read_as_points():
    shield = FaceShield()
    out = shield.transform_one({'a': 1.0, 'b': 2.0, 'c': 3.0})
    assert out['points'].shape == (3, 2)
    assert np.allclose(out['points'][:, 0], [1.0, 2.0, 3.0])
