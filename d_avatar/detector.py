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

    def _to_mp_image(self, image: Any) -> Any:
        """Convert numpy array to MediaPipe Image if needed."""
        if isinstance(image, np.ndarray):
            import mediapipe as mp  # type: ignore[import-untyped]  # noqa: PLC0415

            return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
        return image

    def detect(self, image: Any, t: float) -> dict:
        """Detect landmarks in ``image`` and return the pipeline frame dict.

        Args:
            image: RGB image as numpy array (H, W, 3) or MediaPipe Image.
            t: timestamp in seconds.
        """
        # Convert numpy array to MediaPipe Image if needed
        mp_image = self._to_mp_image(image)

        result = self._landmarker.detect(mp_image)
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
        # MediaPipe landmarks are normalized [0,1], scale to image pixels
        w, h = mp_image.width, mp_image.height
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
    def open_default(cls, model_path: str = 'face_landmarker.task') -> FaceDetector:
        """Build the default MediaPipe ``FaceLandmarker`` based detector.

        ``mediapipe`` is imported only here.

        Args:
            model_path: path to the face_landmarker.task model file.
        """
        from mediapipe.tasks.python import BaseOptions  # type: ignore[import-untyped]  # noqa: I001,PLC0415
        from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]  # noqa: I001,PLC0415
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (  # type: ignore[import-untyped]  # noqa: PLC0415
            VisionTaskRunningMode,
        )
        import os  # noqa: PLC0415  # intentional lazy import

        if not os.path.exists(model_path):
            # Try to find it in mediapipe package
            import mediapipe as mp  # noqa: PLC0415  # intentional lazy import

            mp_root = os.path.dirname(mp.__file__)
            candidate = os.path.join(mp_root, 'modules', 'face_landmarker', 'face_landmarker.task')
            if os.path.exists(candidate):
                model_path = candidate

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                'face_landmarker.task model file not found. '
                'Download from https://storage.googleapis.com/mediapipe-models/'
                'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = FaceLandmarker.create_from_options(options)
        return cls(landmarker)
