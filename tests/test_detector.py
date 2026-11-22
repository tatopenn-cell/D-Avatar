"""Tests for detector.py (item D): a landmarker wrapped into pipeline frames.

Runs without mediapipe installed: mediapipe is imported only inside
FaceDetector.open_default. The landmarker interface mirrors MediaPipe's
FaceLandmarkerResult (``face_landmarks`` list of faces of landmarks with
normalized ``x``/``y``).
"""

import numpy as np

from d_avatar.detector import FaceDetector
from d_avatar.mesh import mesh_to_68
from d_avatar.shield import FaceShield


class _Lm:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.z = 0.0


class _FakeResult:
    def __init__(self, faces):
        self.face_landmarks = faces


class _FakeLandmarker:
    def __init__(self, mesh) -> None:
        self.mesh = mesh

    def detect(self, image):
        h, w = np.asarray(image).shape[:2]
        if self.mesh is None:
            return _FakeResult([])
        faces = [[_Lm(float(x) / w, float(y) / h) for (x, y) in self.mesh]]
        return _FakeResult(faces)


def _mesh_468() -> np.ndarray:
    """A minimal mesh with key points for testing."""
    mesh = np.zeros((468, 2), dtype=float)
    # Eye corners (outer/inner) for both eyes
    mesh[33] = [20.0, 35.0]  # RIGHT_EYE outer (temple) -> iBUG 45
    mesh[133] = [30.0, 35.0]  # RIGHT_EYE inner (nose) -> iBUG 42
    mesh[362] = [70.0, 35.0]  # LEFT_EYE inner (nose) -> iBUG 39
    mesh[263] = [80.0, 35.0]  # LEFT_EYE outer (temple) -> iBUG 36
    return mesh


def test_detector_returns_mesh_frame_with_confidence_one():
    mesh = _mesh_468()
    landmarker = _FakeLandmarker(mesh)
    det = FaceDetector(landmarker)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    frame = det.detect(image, t=1.5)
    assert frame['t'] == 1.5
    assert frame['points'].shape == (68, 2)
    # Reference computed by mesh_to_68 (not by detector)
    ref = mesh_to_68(mesh)
    assert np.allclose(frame['points'], ref, equal_nan=True)
    assert np.all(frame['confidence'] == 1.0)
    assert frame['iris'].shape == (2, 2)


def test_detector_no_face_gives_zero_confidence():
    landmarker = _FakeLandmarker(None)
    det = FaceDetector(landmarker)
    frame = det.detect(np.zeros((10, 10, 3), dtype=np.uint8), t=0.0)
    assert np.all(frame['points'] == 0.0)
    assert np.all(frame['confidence'] == 0.0)
    assert np.all(frame['iris'] == 0.0)


def test_no_face_frame_holds_previous_point_under_shield():
    mesh = _mesh_468()
    det_face = FaceDetector(_FakeLandmarker(mesh))
    det_none = FaceDetector(_FakeLandmarker(None))
    shield = FaceShield(window=7, n_sigmas=3.0, max_gap=5)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    face_frame = det_face.detect(image, t=0.0)
    with_face = shield.transform_one(face_frame)
    noface_frame = det_none.detect(image, t=0.033)
    assert np.all(noface_frame['confidence'] == 0.0)
    without = shield.transform_one(noface_frame)
    # Predict-only during one-frame occlusion keeps the point close.
    assert np.allclose(without['points'], with_face['points'], atol=1.0)
