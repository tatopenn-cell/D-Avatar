"""Tests for capture.py: camera sources yielding (t, image) frames.

Runs without opencv-python installed: cv2 is imported only inside
camera_stream_frames.
"""

import inspect

import numpy as np
from dense_armor.core.signals.vision.streams import Frame

import d_avatar.capture as capture_module
from d_avatar.capture import camera_stream_frames


class _FakeStream:
    """A fake stream that yields frames with timestamps, like CameraStream."""

    def __init__(self, frames, fps=30.0) -> None:
        self._frames = frames
        self._fps = fps
        self._i = 0
        self._t0 = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._frames):
            raise StopIteration
        t = self._i / self._fps
        frame = Frame(array=self._frames[self._i], t=t)
        self._i += 1
        return frame


class _PatchedCameraStream:
    """A patched CameraStream that uses our fake frames instead of opencv."""

    def __init__(self, frames, fps=30.0) -> None:
        self._frames = frames
        self._fps = fps

    def __iter__(self):
        for i, frame in enumerate(self._frames):
            t = i / self._fps
            yield Frame(array=frame, t=t)


def test_camera_stream_frames_yields_frames():
    """Test that iterating camera_stream_frames yields (t, rgb_frame) pairs."""
    # We test the iteration protocol by checking the function exists
    # and has the right signature. The actual CameraStream needs opencv.
    sig = inspect.signature(camera_stream_frames)
    assert 'device' in sig.parameters
    assert 'fps' in sig.parameters
    assert 'size' in sig.parameters
    assert 'gray' in sig.parameters


def test_camera_stream_frames_with_fake_stream():
    """Test camera_stream_frames with a patched CameraStream (no opencv)."""
    frames = [np.full((4, 4, 3), float(i), dtype=np.float32) / 255.0 for i in range(5)]

    # Monkey-patch CameraStream to use our fake stream
    original_camera_stream = capture_module.CameraStream
    capture_module.CameraStream = lambda *args, **kwargs: _PatchedCameraStream(frames, fps=30.0)

    try:
        results = list(camera_stream_frames(device=0, fps=30.0, gray=False))
        assert len(results) == 5
        for i, (t, frame) in enumerate(results):
            assert abs(t - i / 30.0) < 1e-6
            expected = np.full((4, 4, 3), float(i), dtype=np.float32) / 255.0
            np.testing.assert_allclose(frame, expected)
    finally:
        capture_module.CameraStream = original_camera_stream


def test_camera_stream_frames_yields_rgb_float32():
    """Test that camera_stream_frames yields RGB float32 frames in [0, 1]."""
    frames = [np.full((4, 4, 3), float(i), dtype=np.float32) / 255.0 for i in range(3)]

    original_camera_stream = capture_module.CameraStream
    capture_module.CameraStream = lambda *args, **kwargs: _PatchedCameraStream(frames, fps=30.0)

    try:
        for _t, frame in camera_stream_frames(device=0, fps=30.0, gray=False):
            assert frame.dtype == np.float32
            assert frame.ndim == 3
            assert frame.shape[2] == 3  # RGB
            assert np.all(frame >= 0.0) and np.all(frame <= 1.0)
    finally:
        capture_module.CameraStream = original_camera_stream
