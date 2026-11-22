"""Face detector: turns a landmarker's output into the frame the pipeline reads.

Source:
- Bazarevsky et al. 2019, *BlazeFace: Sub-millisecond Neural Face Detection on
  Mobile GPUs*, arXiv:1907.05047: the face detector that gives the region of
  interest.
- Kartynnik et al. 2019, section 5 (page 3): landmark jitter removed with a
  temporal filter.

``FaceDetector`` takes a landmarker object, calls it on an image and returns
the frame dict the pipeline reads::

    {"points": (68, 2), "confidence": (68,), "t": float, "iris": (2, 2)}

When the landmarker finds no face the frame has ``points`` all zero,
``confidence`` all zero and ``iris`` zero: this is the lost-tracking input
that :class:`d_avatar.shield.FaceShield` holds the previous point for.

``mediapipe`` is imported only by :func:`open_default`, which builds the
default :class:`mediapipe.tasks.vision.FaceLandmarker`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from d_avatar.mesh import iris_centres, mesh_to_68


class FaceDetector:
    """Wrap a landmarker and emit pipeline frames.

    Args:
        landmarker: an object with a ``detect(image)`` method returning a
            result with a ``face_landmarks`` attribute (a list of faces, each
            a list of landmarks with ``x``/``y`` normalized coordinates, as
            MediaPipe's ``FaceLandmarkerResult``). An empty list means no
            face.
    """

    def __init__(self, landmarker: Any) -> None:
        self._landmarker = landmarker

    def detect(self, image: Any, t: float) -> dict:
        """Detect landmarks in ``image`` and return the pipeline frame dict."""
        result = self._landmarker.detect(image)
        faces = getattr(result, 'face_landmarks', None) or []
        t = float(t)
        if not faces:
            return {
                'points': np.zeros((68, 2), dtype=float),
                'confidence': np.zeros(68, dtype=float),
                't': t,
                'iris': np.zeros((2, 2), dtype=float),
            }
        landmarks = faces[0]
        arr = np.asarray(image)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        mesh = np.array([[float(lm.x) * w, float(lm.y) * h] for lm in landmarks], dtype=float)
        points = mesh_to_68(mesh)
        confidence: np.ndarray = np.ones(68, dtype=float)
        iris = iris_centres(mesh)
        return {
            'points': points,
            'confidence': confidence,
            't': t,
            'iris': iris,
        }

    @classmethod
    def open_default(cls) -> FaceDetector:
        """Build the default MediaPipe ``FaceLandmarker`` based detector.

        ``mediapipe`` is imported only here.
        """
        from mediapipe.tasks.python.core import BaseOptions  # type: ignore[import-untyped]  # noqa: PLC0415,I001
        from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]  # noqa: PLC0415,I001
            FaceLandmarker,  # noqa: PLC0415
            FaceLandmarkerOptions,  # noqa: PLC0415
        )  # noqa: PLC0415

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=__import__(
                'mediapipe.tasks.python.vision.core.vision_task_running_mode',
                fromlist=['VisionTaskRunningMode'],
            ).VisionTaskRunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = FaceLandmarker.create_from_options(options)
        return cls(landmarker)
