"""Camera source: yields frames with timestamps from a capture object.

The capture object is passed in as an argument; ``opencv-python`` is imported
only by :func:`open_default`, which builds the default capture. A class that
needs ``cv2`` takes the capture object as an argument so tests run without
``opencv-python`` installed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any


class CameraSource:
    """Yield ``(t, image)`` frames from a capture until it reports no frame.

    Args:
        cap: a capture object with a ``read() -> (ok, image)`` method.
        clock: clock callable returning seconds (default ``time.perf_counter``).
    """

    def __init__(self, cap: Any, clock: Callable[[], float] = time.perf_counter) -> None:
        self._cap = cap
        self._clock = clock

    def frames(self) -> Iterator[tuple[float, Any]]:
        """Yield ``(t, image)`` until the capture reports no frame."""
        while True:
            ok, image = self._cap.read()
            if not ok:
                return
            yield (self._clock(), image)

    def close(self) -> None:
        """Release the capture."""
        self._cap.release()

    @classmethod
    def open_default(cls, index: int = 0) -> CameraSource:
        """Build a :class:`CameraSource` from the default camera.

        ``opencv-python`` is imported only here.
        """
        import cv2  # noqa: PLC0415  # intentional lazy import

        cap = cv2.VideoCapture(index)
        return cls(cap)
