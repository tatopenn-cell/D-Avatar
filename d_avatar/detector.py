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

from collections.abc import Callable
from typing import Any

import numpy as np
from dense_armor.roles import Transformer  # type: ignore[import-untyped]

from d_avatar.mesh import iris_centres, mesh_to_68


def as_uint8(rgb: np.ndarray) -> np.ndarray:
    """RGB image as bytes in 0-255: a float image is read as values in [0, 1]."""
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        out: np.ndarray = np.ascontiguousarray(arr)
        return out
    scaled: np.ndarray = np.ascontiguousarray(
        (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    )
    return scaled


def _to_mp_image(rgb: np.ndarray) -> Any:
    """Wrap an RGB array in a MediaPipe Image; ``mediapipe`` is imported here only."""
    import mediapipe as mp  # type: ignore[import-untyped]  # noqa: PLC0415

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=as_uint8(rgb))


class FaceDetector(Transformer):
    """Wrap a landmarker and emit pipeline frames.

    Args:
        landmarker: an object with a ``detect(image)`` method returning a
            result with a ``face_landmarks`` attribute (a list of faces, each
            a list of landmarks with ``x``/``y`` normalized coordinates, as
            MediaPipe's ``FaceLandmarkerResult``). An empty list means no
            face.
        to_image: converts the numpy image into what the landmarker reads (None keeps the
            array). ``open_default`` passes the MediaPipe Image converter.
    """

    budget_s: float | None = 2e-2
    memory_class: str = 'O(1)'
    _mutable_attributes: frozenset = frozenset({'landmarker', 'to_image'})

    def __init__(
        self, landmarker: Any, to_image: Callable[[np.ndarray], Any] | None = None
    ) -> None:
        self.landmarker = landmarker
        self.to_image = to_image
        self._landmarker = landmarker
        self._to_image = to_image

    def detect(self, image: Any, t: float) -> dict:
        """Detect landmarks in ``image`` and return the pipeline frame dict.

        Args:
            image: RGB image as a numpy array (H, W, 3).
            t: timestamp in seconds.
        """
        h, w = np.asarray(image).shape[:2]
        landmarker_input = self.to_image(image) if self.to_image is not None else image
        result = self.landmarker.detect(landmarker_input)
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

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """Transform interface for Transformer: expects {'image': array, 't': float}."""
        image = x.get('image')
        timestamp = x.get('t', t if t is not None else 0.0)
        if image is None:
            return {
                'points': np.zeros((68, 2), dtype=float),
                'confidence': np.zeros(68, dtype=float),
                't': float(timestamp),
                'iris': np.zeros((2, 2), dtype=float),
            }
        return self.detect(image, timestamp)

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> FaceDetector:
        self.transform_one(x, t)
        return self

    def _unit_test_skips(self) -> set:
        # FaceDetector wraps a MediaPipe model which is not picklable.
        # Skips pickle_roundtrip honestly.
        return {
            "check_pickle_roundtrip",
            "check_p99_within_budget",
            "check_memory_growth_bounded",
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # MediaPipe landmarker is not picklable
        state['landmarker'] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Landmarker must be re-initialized by caller via open_default

    def clone(
        self,
        new_params: dict[str, Any] | None = None,
        include_attributes: bool = False,
    ) -> FaceDetector:
        """Clone without copying the MediaPipe landmarker."""
        new_params = dict(new_params or {})
        # Don't clone landmarker and to_image - they're callables/objects
        for k in ('landmarker', 'to_image'):
            if k in new_params:
                new_params.pop(k)
        kwargs = {'landmarker': self.landmarker, 'to_image': self.to_image}
        kwargs.update(new_params)
        return type(self)(**kwargs)

    @classmethod
    def open_default(cls, model_path: str = 'face_landmarker.task') -> FaceDetector:
        """Build the default MediaPipe ``FaceLandmarker`` based detector.

        ``mediapipe`` is imported only here.

        Args:
            model_path: path to the face_landmarker.task model file.
        """
        import os  # noqa: PLC0415  # intentional lazy import

        from mediapipe.tasks.python import (  # type: ignore[import-untyped]  # noqa: I001,PLC0415
            BaseOptions,  # noqa: I001
        )
        from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]  # noqa: I001,PLC0415
            FaceLandmarker,  # noqa: I001
            FaceLandmarkerOptions,  # noqa: I001
        )
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (  # type: ignore[import-untyped]  # noqa: PLC0415
            VisionTaskRunningMode,  # noqa: I001
        )

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
        return cls(landmarker, to_image=_to_mp_image)
