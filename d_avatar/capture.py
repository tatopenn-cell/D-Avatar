"""Camera source: yield frames with timestamps from Dense-Armor's CameraStream.

``opencv-python`` is imported only by :func:`camera_stream_frames`.
Tests use a fake stream with ``camera_stream_frames``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from dense_armor.core.signals.vision.streams import CameraStream  # type: ignore[import-untyped]


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
