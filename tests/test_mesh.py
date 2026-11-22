"""Tests for mesh_to_68 / iris_centres (item A).

The synthetic 468-point mesh has a known geometry so the corner and lid
points can be computed by hand. The canonical MediaPipe neutral mesh is the
documented test input; it is loaded from the installed mediapipe package when
``canonical_face_model.obj`` is available, otherwise that sub-test is skipped
(the source file is not part of the published mediapipe wheel and could not
be fetched).
"""

from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from d_avatar.mesh import (
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    _contour_loop,
    iris_centres,
    mesh_to_68,
)
from d_avatar.params import eye_aspect_ratios, landmarks_to_params

_TOL = 1e-6


def _synthetic_mesh_468() -> np.ndarray:
    """Neutral 468-point mesh in MediaPipe convention.

    Image-left eye contour -> iBUG 36-41 (ellipse centre (30, 35), rx 5, ry 3),
    image-right eye contour -> iBUG 42-47 (centre (70, 35)). Nose centred at
    x = 50 so head_yaw is 0; closed mouth so mouth_open is ~0.
    """
    mesh = np.zeros((468, 2), dtype=float)
    # Face contours
    for contour, cx in ((FACEMESH_LEFT_EYE, 30.0), (FACEMESH_RIGHT_EYE, 70.0)):
        loop = _contour_loop(contour)
        for p, idx in enumerate(loop):
            theta = 2.0 * np.pi * p / len(loop)
            mesh[idx] = [cx + 5.0 * np.cos(theta), 35.0 + 3.0 * np.sin(theta)]
    for idx in sorted({i for e in FACEMESH_NOSE for i in e}):
        mesh[idx] = [50.0, 40.0]
    # Closed mouth: all lip vertices at one point -> mouth_open ~ 0.
    for idx in sorted({i for e in FACEMESH_LIPS for i in e}):
        mesh[idx] = [50.0, 70.0]
    for contour, cx in ((FACEMESH_LEFT_EYEBROW, 30.0), (FACEMESH_RIGHT_EYEBROW, 70.0)):
        loop = _contour_loop(contour)
        for p, idx in enumerate(loop):
            theta = 2.0 * np.pi * p / len(loop)
            mesh[idx] = [cx + 4.0 * np.cos(theta), 25.0 + 1.5 * np.sin(theta)]
    return mesh


def _synthetic_mesh_478() -> np.ndarray:
    """The 468 neutral mesh plus the 10 refined iris vertices (469-477)."""
    mesh = _synthetic_mesh_468()
    mesh = np.vstack([mesh, np.zeros((10, 2), dtype=float)])
    # Image-left eye (iBUG left) uses MediaPipe LEFT_IRIS indices 474-477;
    # image-right eye uses RIGHT_IRIS indices 469-472.
    for idx in range(474, 478):
        mesh[idx] = [30.0, 35.0]
    for idx in range(469, 473):
        mesh[idx] = [70.0, 35.0]
    return mesh


def test_mesh_wrong_shape_raises():
    with pytest.raises(ValueError):
        mesh_to_68(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        mesh_to_68(np.zeros((468, 3))[..., :1])
    with pytest.raises(ValueError):
        iris_centres(np.zeros((5, 2)))


def test_synthetic_mesh_known_geometry():
    mesh = _synthetic_mesh_468()
    pts = mesh_to_68(mesh)
    assert pts.shape == (68, 2)
    # Corners: outer = min-x corner (temple), inner = max-x corner (nose).
    # Image-left eye (iBUG left) at x ~30: outer=36 at ~25, inner=39 at ~35.
    assert np.allclose(pts[36], [25.0, 35.0], atol=_TOL)
    assert np.allclose(pts[39], [35.0, 35.0], atol=_TOL)
    # Image-right eye (iBUG right) at x ~70: inner=42 at ~75 (nose), outer=45 at ~65 (temple).
    assert np.allclose(pts[42], [75.0, 35.0], atol=_TOL)
    assert np.allclose(pts[45], [65.0, 35.0], atol=_TOL)
    # Lid points: upper lids above the corners, lower lids below.
    assert pts[37, 1] < 35.0 and pts[38, 1] < 35.0
    assert pts[40, 1] > 35.0 and pts[41, 1] > 35.0
    assert pts[43, 1] < 35.0 and pts[44, 1] < 35.0
    assert pts[46, 1] > 35.0 and pts[47, 1] > 35.0
    # Jaw (0-16) has no mesh counterpart: NaN, never read.
    assert np.all(np.isnan(pts[0:17]))


def test_synthetic_mesh_parameters():
    mesh = _synthetic_mesh_468()
    ear_l, ear_r = eye_aspect_ratios(mesh_to_68(mesh))
    ear_open = (ear_l + ear_r) / 2.0
    p = landmarks_to_params(mesh_to_68(mesh), ear_open=ear_open)
    assert abs(p['eye_l'] - 1.0) < 0.05
    assert abs(p['eye_r'] - 1.0) < 0.05
    assert p['mouth_open'] < 0.05
    assert abs(p['head_roll']) < 1.0
    assert abs(p['head_yaw']) < 5.0


def _load_canonical_mesh():
    # Try to find canonical_face_model.obj in the mediapipe package
    try:
        mp_root = Path(resources.files('mediapipe'))
        obj_path = mp_root / 'modules' / 'face_landmark' / 'canonical_face_model.obj'
        if not obj_path.exists():
            return None
    except Exception:
        return None

    verts = []
    with obj_path.open(encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('v '):
                _, x, y, _z = line.split(maxsplit=3)[:4]
                verts.append((float(x), float(y)))
    if not verts:
        return None
    return np.array(verts, dtype=float)


def test_canonical_mesh_parameters():
    mesh = _load_canonical_mesh()
    if mesh is None:
        pytest.skip('canonical_face_model.obj not available in mediapipe package')
    assert mesh.shape[0] in (468, 478)
    pts = mesh_to_68(mesh)
    ear_l, ear_r = eye_aspect_ratios(pts)
    ear_open = (ear_l + ear_r) / 2.0
    p = landmarks_to_params(pts, ear_open=ear_open)
    assert abs(p['eye_l'] - 1.0) < 0.05
    assert abs(p['eye_r'] - 1.0) < 0.05
    assert p['mouth_open'] < 0.05
    assert abs(p['head_roll']) < 1.0
    assert abs(p['head_yaw']) < 5.0


def test_iris_centres():
    mesh = _synthetic_mesh_478()
    c = iris_centres(mesh)
    assert c.shape == (2, 2)
    # Left iris (image-left eye) at x < right iris (image-right eye).
    assert c[0, 0] < c[1, 0]
    # 468-point mesh has no iris vertices -> zero centres.
    mesh468 = np.zeros((468, 2), dtype=float)
    assert np.allclose(iris_centres(mesh468), 0.0)


def test_iris_mapping_consistent_with_contours():
    """RIGHT_EYE contour and RIGHT_IRIS should map to same iBUG eye."""
    mesh = _synthetic_mesh_478()
    # Image-left eye at x~30 uses LEFT_EYE contour and LEFT_IRIS (474-477)
    # Image-right eye at x~70 uses RIGHT_EYE contour and RIGHT_IRIS (469-472)
    c = iris_centres(mesh)
    # Both centres should be on their respective eye contours
    assert abs(c[0, 0] - 30.0) < 1.0  # left iris near x=30
    assert abs(c[1, 0] - 70.0) < 1.0  # right iris near x=70
