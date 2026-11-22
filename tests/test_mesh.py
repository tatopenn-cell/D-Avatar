"""Tests for mesh_to_68 / iris_centres (item A).

The synthetic 468-point mesh has a known geometry so the corner and lid
points can be computed by hand. The canonical MediaPipe neutral mesh is the
documented test input; it is loaded from the installed mediapipe package when
``canonical_face_model.obj`` is available, otherwise that sub-test is skipped
(the source file is not part of the published mediapipe wheel and could not
be fetched).
"""

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

    Normal (non-mirrored) camera image convention:
    - FACEMESH_LEFT_EYE at x=70 (person's LEFT eye, iBUG 36-41)
    - FACEMESH_RIGHT_EYE at x=30 (person's RIGHT eye, iBUG 42-47)
    Nose centred at x=50 so head_yaw is 0; closed mouth so mouth_open is ~0.
    """
    mesh = np.zeros((468, 2), dtype=float)
    # Face contours: LEFT_EYE at x=70 (person's LEFT), RIGHT_EYE at x=30 (person's RIGHT)
    for contour, cx in ((FACEMESH_LEFT_EYE, 70.0), (FACEMESH_RIGHT_EYE, 30.0)):
        loop = _contour_loop(contour)
        for p, idx in enumerate(loop):
            theta = 2.0 * np.pi * p / len(loop)
            mesh[idx] = [cx + 5.0 * np.cos(theta), 35.0 + 3.0 * np.sin(theta)]
    for idx in sorted({i for e in FACEMESH_NOSE for i in e}):
        mesh[idx] = [50.0, 40.0]
    # Closed mouth: all lip vertices at one point -> mouth_open ~ 0.
    for idx in sorted({i for e in FACEMESH_LIPS for i in e}):
        mesh[idx] = [50.0, 70.0]
    for contour, cx in ((FACEMESH_LEFT_EYEBROW, 70.0), (FACEMESH_RIGHT_EYEBROW, 30.0)):
        loop = _contour_loop(contour)
        for p, idx in enumerate(loop):
            theta = 2.0 * np.pi * p / len(loop)
            mesh[idx] = [cx + 4.0 * np.cos(theta), 25.0 + 1.5 * np.sin(theta)]
    return mesh


def _synthetic_mesh_478() -> np.ndarray:
    """The 468 neutral mesh plus the 10 refined iris vertices (469-477)."""
    mesh = _synthetic_mesh_468()
    mesh = np.vstack([mesh, np.zeros((10, 2), dtype=float)])
    # Normal convention: LEFT_IRIS (474-477) at x=70 (person's LEFT eye)
    # RIGHT_IRIS (469-472) at x=30 (person's RIGHT eye)
    for idx in range(474, 478):
        mesh[idx] = [70.0, 35.0]
    for idx in range(469, 473):
        mesh[idx] = [30.0, 35.0]
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
    # Corners: outer = temple side, inner = nose side.
    # DA1 convention: indices 36-41 = person's RIGHT eye, 42-47 = person's LEFT eye.
    # Synthetic mesh: FACEMESH_LEFT_EYE at x=70 (person's LEFT eye),
    # FACEMESH_RIGHT_EYE at x=30 (person's RIGHT eye).
    # DA1 convention puts RIGHT eye at 36-41, LEFT eye at 42-47.
    # Person's RIGHT eye (iBUG 36-41) at x~30: outer=36 at min x (25, temple),
    # inner=39 at max x (35, nose).
    assert np.allclose(pts[36], [25.0, 35.0], atol=_TOL)  # right outer (temple)
    assert np.allclose(pts[39], [35.0, 35.0], atol=_TOL)  # right inner (nose)
    # Person's LEFT eye (iBUG 42-47) at x~70: inner=42 at min x (65, nose),
    # outer=45 at max x (75, temple).
    assert np.allclose(pts[42], [65.0, 35.0], atol=_TOL)  # left inner (nose)
    assert np.allclose(pts[45], [75.0, 35.0], atol=_TOL)  # left outer (temple)
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
    assert abs(p['head_roll']) < 1.0
    assert abs(p['head_yaw']) < 5.0


_CANONICAL = Path(__file__).resolve().parents[1] / 'papers' / 'canonical_face_model.obj'
_LOWER_LIP = [17, 314, 405, 321, 375, 84, 181, 91, 146, 14, 317, 402, 318, 324, 87, 178, 88, 95]


def _canonical_vertices():
    """The 468 vertices (x, y, z) of the canonical face model, y pointing up."""
    if not _CANONICAL.exists():
        pytest.skip('papers/canonical_face_model.obj not available')
    rows = [
        ln.split()[1:4]
        for ln in _CANONICAL.read_text(encoding='utf-8').splitlines()
        if ln.startswith('v ')
    ]
    verts = np.array([[float(v) for v in r] for r in rows if len(r) == 3])
    assert verts.shape == (468, 3)
    return verts


def _image_mesh(verts):
    return np.column_stack([verts[:, 0], -verts[:, 1]])


def _params(mesh):
    pts = mesh_to_68(mesh)
    ear_open = float(np.mean(eye_aspect_ratios(pts)))
    return pts, landmarks_to_params(pts, ear_open=ear_open)


@pytest.mark.parametrize('mirror', [1.0, -1.0])
def test_canonical_mesh_neutral_parameters(mirror):
    mesh = _image_mesh(_canonical_vertices()) * [mirror, 1.0]
    pts, p = _params(mesh)
    assert abs(p['eye_l'] - 1.0) < 0.05
    assert abs(p['eye_r'] - 1.0) < 0.05
    assert abs(p['head_roll']) < 1.0
    assert abs(p['head_yaw']) < 5.0
    xs = pts[[36, 39, 42, 45], 0]
    assert list(xs) == sorted(xs)


def test_canonical_mouth_width_and_opening():
    mesh = _image_mesh(_canonical_vertices())
    pts, neutral = _params(mesh)
    width = float(np.linalg.norm(mesh[61] - mesh[291]))
    assert abs(float(np.linalg.norm(pts[48] - pts[54])) - width) < 0.05 * width
    inner_gap = float(np.linalg.norm(mesh[13] - mesh[14]))
    assert abs(neutral['mouth_open'] - inner_gap / width / 0.5) < 0.05
    opened = mesh.copy()
    opened[_LOWER_LIP, 1] += 0.35 * width
    assert _params(opened)[1]['mouth_open'] > 0.5 > neutral['mouth_open']


def test_canonical_head_yaw_grows_with_rotation():
    verts = _canonical_vertices()
    yaws = []
    for deg in (0.0, 20.0, 40.0):
        a = np.radians(deg)
        x = verts[:, 0] * np.cos(a) + verts[:, 2] * np.sin(a)
        yaws.append(_params(np.column_stack([x, -verts[:, 1]]))[1]['head_yaw'])
    assert yaws[0] < yaws[1] < yaws[2]


def test_iris_centres():
    mesh = _synthetic_mesh_478()
    c = iris_centres(mesh)
    assert c.shape == (2, 2)
    # Returns [left_iris, right_iris] in iBUG order:
    # left_iris = person's LEFT eye (iBUG 36-41) at higher x in non-mirrored image
    # right_iris = person's RIGHT eye (iBUG 42-47) at lower x
    assert c[0, 0] > c[1, 0]
    # 468-point mesh has no iris vertices -> zero centres.
    mesh468 = np.zeros((468, 2), dtype=float)
    assert np.allclose(iris_centres(mesh468), 0.0)


def test_iris_mapping_consistent_with_contours():
    """RIGHT_EYE contour and RIGHT_IRIS should map to same iBUG eye."""
    mesh = _synthetic_mesh_478()
    # Image-left eye at x~30 uses LEFT_EYE contour and LEFT_IRIS (474-477)
    # Image-right eye at x~70 uses RIGHT_EYE contour and RIGHT_IRIS (469-472)
    c = iris_centres(mesh)
    # c[0] = left_iris = person's LEFT eye (iBUG 36-41) at x~70
    # c[1] = right_iris = person's RIGHT eye (iBUG 42-47) at x~30
    assert abs(c[0, 0] - 70.0) < 1.0  # left iris near x=70
    assert abs(c[1, 0] - 30.0) < 1.0  # right iris near x=30
