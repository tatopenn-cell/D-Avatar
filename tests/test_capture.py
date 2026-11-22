"""Tests for capture.py (item C): a camera source yielding (t, image) frames.

Runs without opencv-python installed: cv2 is imported only inside
CameraSource.open_default.
"""

import numpy as np

from d_avatar.capture import CameraSource


class _FakeCap:
    """A capture that yields `n` frames then reports a failed read."""

    def __init__(self, n: int = 5) -> None:
        self.n = int(n)
        self.i = 0
        self.released = False

    def read(self):
        if self.i >= self.n:
            return False, None
        frame = np.full((4, 4, 3), float(self.i), dtype=np.uint8)
        self.i += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_frames_yield_relative_timestamps_starting_near_zero():
    """Timestamps should start near 0 and be non-decreasing."""
    cap = _FakeCap(5)
    src = CameraSource(cap, clock=lambda: 0.0)
    frames = list(src.frames())
    assert len(frames) == 5
    ts = [t for t, _ in frames]
    # First timestamp should be ~0 (relative to start)
    assert abs(ts[0]) < 1e-6
    # Non-decreasing
    assert all(b >= a for a, b in zip(ts, ts[1:], strict=False))


def test_frames_stop_on_failing_read():
    cap = _FakeCap(3)
    src = CameraSource(cap, clock=lambda: 0.0)
    frames = list(src.frames())
    assert len(frames) == 3


def test_close_releases_capture():
    cap = _FakeCap(5)
    src = CameraSource(cap)
    src.close()
    assert cap.released is True


def test_frames_with_advancing_clock():
    """Timestamps advance with the clock."""
    now = [0.0]

    def clock():
        result = now[0]
        now[0] += 0.033
        return result

    cap = _FakeCap(5)
    src = CameraSource(cap, clock=clock)
    frames = list(src.frames())
    ts = [t for t, _ in frames]
    assert abs(ts[0]) < 1e-6
    # Differences should be ~0.033
    diffs = [b - a for a, b in zip(ts, ts[1:], strict=False)]
    assert all(abs(d - 0.033) < 1e-3 for d in diffs)
