"""Gaze from the iris: pupil displacement turned into blend shapes in [-1, 1].

Source:
- Ablavatski et al. 2020, *Real-time Pupil Tracking from Monocular Video for
  Digital Puppetry*, arXiv:2006.11341, section 3 (page 2): for every pupil
  blend shape the displacement ``D_current`` between two mesh vertices is
  compared to ``D_neutral`` (minimum activation) and ``D_activated`` (maximum
  activation), giving a scalar in [0, 1] per blend shape; pairs of opposite
  blend shapes are merged into two aggregate values.
- Section 3.1 (page 3): real-time calibration keeps a circular buffer of
  trusted displacements; the calibrated displacement is the mean of the
  trusted displacements and their standard deviation is the confidence interval
  (Algorithm 1, Standard Score Filter).

``iris_displacement`` returns the raw (dx, dy) displacement of the iris from
the eye corners, normalised by the eye geometry. ``GazeCalibrator`` tracks the
calibrated neutral displacement in a circular buffer of trusted displacements
and maps a new displacement to ``gaze_x, gaze_y`` in [-1, 1]: 0 at the neutral
displacement, +1 / -1 at the activated displacement.

The paper gives no number for ``D_activated``. ``D_ACTIVATED`` below is a
declared constant of this implementation, not a calibrated value.
"""

from __future__ import annotations

import numpy as np

# Declared constant, not calibrated: the paper does not give a value for
# D_activated. In units of the normalised displacement returned by
# iris_displacement, full activation is +/-1.
D_ACTIVATED = 1.0
_N_SIGMAS = 3.0
# Minimum standard deviation used for the confidence interval. The paper's
# variance starts at zero, which would reject every new displacement; this
# floor lets the calibrator adapt to a sustained shift while still rejecting
# genuine outliers. A declared constant of this implementation.
_MIN_STD = 0.1


def iris_displacement(
    iris: np.ndarray,
    corner_in: np.ndarray,
    corner_out: np.ndarray,
    lid_up: np.ndarray,
    lid_down: np.ndarray,
) -> tuple[float, float]:
    """Normalised displacement of the iris from the eye corners.

    Args:
        iris: iris centre (x, y).
        corner_in: inner eye corner (nose side) (x, y).
        corner_out: outer eye corner (temple side) (x, y).
        lid_up: upper eyelid point (x, y).
        lid_down: lower eyelid point (x, y).

    Returns:
        (dx, dy) normalised displacement. ``dx`` is +1 when the iris sits on
        the outer corner and -1 on the inner corner; ``dy`` is +1 when the iris
        sits on the lower lid and -1 on the upper lid. The neutral (frontally
        looking) eye gives approximately (0, 0).
    """
    iris = np.asarray(iris, dtype=float)
    ci = np.asarray(corner_in, dtype=float)
    co = np.asarray(corner_out, dtype=float)
    lu = np.asarray(lid_up, dtype=float)
    ld = np.asarray(lid_down, dtype=float)
    eye_w = float(np.linalg.norm(co - ci)) + 1e-9
    mid = (ci + co) / 2.0
    dx = float((iris[0] - mid[0]) / (eye_w / 2.0))
    lid_mid = (lu + ld) / 2.0
    eye_h = float(np.linalg.norm(ld - lu)) + 1e-9
    dy = float((iris[1] - lid_mid[1]) / (eye_h / 2.0))
    return dx, dy


class GazeCalibrator:
    """Real-time calibrator of the neutral iris displacement.

    Keeps a circular buffer of ``size`` trusted displacements. A new
    displacement is trusted when it falls within the confidence interval
    (mean +/- ``n_sigmas`` * std of the trusted displacements); trusted
    displacements are inserted into the circular buffer, evicting the oldest.
    The calibrated neutral displacement is the mean of the buffer.

    Args:
        size: circular buffer length (default 60).
        n_sigmas: half-width of the confidence interval in std units.
    """

    def __init__(self, size: int = 60, n_sigmas: float = _N_SIGMAS) -> None:
        self.size = int(size)
        self.n_sigmas = float(n_sigmas)
        self._buf: np.ndarray = np.zeros((self.size, 2), dtype=float)
        self._i: int = 0
        self._n: int = 0
        self._mean: np.ndarray = np.zeros(2, dtype=float)
        self._std: np.ndarray = np.ones(2, dtype=float)

    def _window(self) -> np.ndarray:
        """The trusted displacements currently in the circular buffer."""
        if self._n == 0:
            return self._buf[:0]
        start = (self._i - self._n) % self.size
        n = self._n
        if start + n <= self.size:
            return self._buf[start : start + n]
        return np.concatenate([self._buf[start:], self._buf[: start + n - self.size]]).astype(float)  # type: ignore[no-any-return]

    def _recompute(self) -> None:
        win = self._window()
        if win.size == 0:
            self._mean = np.zeros(2, dtype=float)
            self._std = np.ones(2, dtype=float)
            return
        self._mean = win.mean(axis=0)
        self._std = np.maximum(win.std(axis=0), _MIN_STD)

    def update(self, dx: float, dy: float) -> None:
        """Feed one displacement; trusted ones enter the circular buffer.

        A displacement is trusted when it falls within the confidence
        interval ``mean +/- n_sigmas * std``. Non-trusted displacements are
        blended toward the current mean (influence ``_ALPHA``) so a single
        outlier cannot jerk the calibrated neutral.
        """
        d = np.array([float(dx), float(dy)], dtype=float)
        if self._n == 0:
            insert = d
        else:
            trusted = bool(np.all(np.abs(d - self._mean) <= self.n_sigmas * self._std + 1e-9))
            insert = d if trusted else self._mean.copy()
        self._buf[self._i] = insert
        self._i = (self._i + 1) % self.size
        self._n = min(self._n + 1, self.size)
        self._recompute()

    def neutral(self) -> tuple[float, float]:
        """Calibrated neutral displacement (mean of the trusted buffer)."""
        return (float(self._mean[0]), float(self._mean[1]))

    def gaze(self, dx: float, dy: float) -> tuple[float, float]:
        """Map a displacement to (gaze_x, gaze_y) in [-1, 1].

        0 at the calibrated neutral displacement, +1 / -1 at the activated
        displacement ``D_ACTIVATED``.
        """
        d = np.array([float(dx), float(dy)], dtype=float)
        gx = float(np.clip((d[0] - self._mean[0]) / D_ACTIVATED, -1.0, 1.0))
        gy = float(np.clip((d[1] - self._mean[1]) / D_ACTIVATED, -1.0, 1.0))
        return gx, gy
