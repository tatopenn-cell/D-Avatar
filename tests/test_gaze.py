"""Tests for gaze.py (item B): iris displacement and GazeCalibrator.

References Ablavatski et al. 2020 section 3 (page 2: displacement-based pupil
blend shapes merged into two aggregates) and section 3.1 (page 3: real-time
calibration with a circular buffer of trusted displacements).
"""

import numpy as np

from d_avatar.gaze import D_ACTIVATED, GazeCalibrator, iris_displacement


def test_iris_displacement_hand_computed():
    # Iris at the eye centre -> neutral displacement (0, 0).
    dx, dy = iris_displacement(
        np.array([50.0, 35.0]),
        np.array([40.0, 35.0]),
        np.array([60.0, 35.0]),
        np.array([50.0, 32.0]),
        np.array([50.0, 38.0]),
    )
    assert abs(dx) < 1e-9
    assert abs(dy) < 1e-9
    # Iris halfway to the outer corner -> dx = +0.5.
    dx, dy = iris_displacement(
        np.array([55.0, 35.0]),
        np.array([40.0, 35.0]),
        np.array([60.0, 35.0]),
        np.array([50.0, 32.0]),
        np.array([50.0, 38.0]),
    )
    assert abs(dx - 0.5) < 1e-9
    assert abs(dy) < 1e-9
    # Iris on the lower lid -> dy = +1.
    dx, dy = iris_displacement(
        np.array([50.0, 38.0]),
        np.array([40.0, 35.0]),
        np.array([60.0, 35.0]),
        np.array([50.0, 32.0]),
        np.array([50.0, 38.0]),
    )
    assert abs(dx) < 1e-9
    assert abs(dy - 1.0) < 1e-9


def test_gaze_neutral_and_activated():
    cal = GazeCalibrator(size=60)
    for _ in range(100):
        cal.update(0.0, 0.0)
    assert cal.neutral() == (0.0, 0.0)
    # 0 at the neutral displacement.
    assert cal.gaze(0.0, 0.0) == (0.0, 0.0)
    # +1 / -1 at the activated displacement.
    assert cal.gaze(D_ACTIVATED, 0.0) == (1.0, 0.0)
    assert cal.gaze(-D_ACTIVATED, 0.0) == (-1.0, 0.0)
    assert cal.gaze(0.0, D_ACTIVATED) == (0.0, 1.0)


def test_gaze_calibrator_outliers_rejected():
    cal = GazeCalibrator(size=60)
    for _ in range(100):
        cal.update(0.0, 0.0)
    # An outlier far outside the confidence interval is not trusted.
    cal.update(100.0, 100.0)
    assert cal.neutral() == (0.0, 0.0)
    assert cal.gaze(0.0, 0.0) == (0.0, 0.0)


def test_circular_buffer_recovers_last_size():
    # Sustained shift inside confidence interval: buffer adapts.
    cal = GazeCalibrator(size=60)
    for _ in range(60):
        cal.update(0.0, 0.0)
    for _ in range(600):
        cal.update(0.2, 0.0)
    nx, ny = cal.neutral()
    assert abs(nx - 0.2) < 1e-6
    assert abs(ny - 0.0) < 1e-6


def test_non_trusted_not_inserted():
    """Non-trusted displacements do not enter the buffer."""
    cal = GazeCalibrator(size=60)
    for _ in range(60):
        cal.update(0.0, 0.0)
    # 100 is far outside confidence interval, should be discarded.
    for _ in range(100):
        cal.update(100.0, 0.0)
    # Neutral should remain near 0.
    nx, ny = cal.neutral()
    assert abs(nx) < 1e-6
    assert abs(ny) < 1e-6
