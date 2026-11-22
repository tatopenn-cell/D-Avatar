"""Camera sources: yield frames with timestamps.

Two sources are provided:
- :class:`CameraSource` wraps any capture object with a ``read()`` method.
  Tests use this with a fake capture; ``opencv-python`` is never imported.
- :func:`camera_stream_frames` uses Dense-Armor's
  :class:`dense_armor.core.signals.vision.streams.CameraStream`
  for real cameras. It yields ``(t, rgb_frame)`` where ``rgb_frame`` is
  float32 in ``[0, 1]`` (RGB order). The app converts to BGR uint8 for display.

``opencv-python`` is imported only by :func:`camera_stream_frames`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

from dense_armor.core.signals.vision.streams import CameraStream  # type: ignore[import-untyped]


class CameraSource:
    """Yield ``(t, image)`` frames from a capture until it reports no frame.

    Args:
        cap: a capture object with a ``read() -> (ok, image)`` method.
        clock: clock callable returning seconds (default ``time.perf_counter``).
    """

    def __init__(self, cap: Any, clock: Callable[[], float] | None = None) -> None:
        self._cap = cap
        self._clock = clock if clock is not None else time.perf_counter
        self._start_t: float | None = None

    def frames(self) -> Iterator[tuple[float, Any]]:
        """Yield ``(t, image)`` until the capture reports no frame.

        Timestamps are relative to the first frame (start at 0).
        """
        while True:
            ok, image = self._cap.read()
            if not ok:
                return
            t = self._clock()
            if self._start_t is None:
                self._start_t = t
            yield (t - self._start_t, image)

    def close(self) -> None:
        """Release the capture."""
        self._cap.release()

    @classmethod
    def open_default(cls, index: int = 0) -> CameraSource:
        """Build a :class:`CameraSource` from the default camera.

        ``opencv-python`` is imported only here.
        """
        import cv2  # noqa: PLC0415  # type: ignore[import-not-found]

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera index {index}')
        return cls(cap)


def camera_stream_frames(
    device: int = 0,
    fps: float = 30.0,
    size: tuple[int, int] | None = (160, 120),
    gray: bool = False,
) -> Iterator[tuple[float, Any]]:
    """Yield ``(t, rgb_frame)`` from Dense-Armor's ``CameraStream``.

    ``rgb_frame`` is float32 in ``[0, 1]``, RGB order. For OpenCV display,
    convert with ``cv2.cvtColor((frame * 255).astype(np.uint8),
    cv2.COLOR_RGB2BGR)``.

    Args:
        device: camera index (default ``0``).
        fps: requested frame rate, in Hz.
        size: ``(width, height)`` to resize each frame to, or ``None``.
        gray: if ``True``, yield grayscale frames (H, W); if ``False``, yield
            RGB frames (H, W, 3).

    Yields:
        Tuples of ``(timestamp_seconds, frame_array)``.

    Raises:
        ImportError: if OpenCV or Dense-Armor vision extras are not installed.
        RuntimeError: if the camera cannot be opened.

    Example:
        >>> for t, rgb in camera_stream_frames(device=0, gray=False):
        ...     bgr = cv2.cvtColor((rgb * 255).astype(np.uint8),
        ...                        cv2.COLOR_RGB2BGR)
        ...     cv2.imshow('cam', bgr)
    """
    stream = CameraStream(device=device, fps=fps, size=size, gray=gray)
    for frame in stream:
        yield (frame.t, frame.array)
