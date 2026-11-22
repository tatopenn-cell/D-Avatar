"""Convert a 468/478-point MediaPipe face mesh to the 68-point iBUG layout.

Source:
- Kartynnik et al. 2019, *Real-time Facial Surface Geometry from Monocular
  Video on Mobile GPUs*, arXiv:1907.06724, section 2: the mesh has 468
  vertices in fixed topology.
- Ablavatski et al. 2020, *Real-time Pupil Tracking from Monocular Video
  for Digital Puppetry*, arXiv:2006.11341, section 2: the refined mesh adds
  ten iris points (five per eye) for a total of 478 vertices.
- The eye, lip, eyebrow and nose vertex sets come from the official MediaPipe
  source `mediapipe/python/solutions/face_mesh_connections.py`
  (the ``FACEMESH_*`` frozensets below), fetched and printed before use.

The iBUG 68-point ordering that ``d_avatar.params.landmarks_to_params`` reads:
indices 17-26 brows, 27-35 nose, 36-47 eyes, 48-67 mouth. Jaw points 0-16 have
no mesh counterpart and are left as NaN (never read).

The eye contours carry no iBUG left/right label. The two contours are assigned
to the iBUG eyes by image x position: the contour with the lower mean x becomes
iBUG 36-41 and the other iBUG 42-47. This keeps the outer eye corners (36 and
45) ordered so that ``head_roll`` (vector 45 - 36) is zero for a level frontal
face, matching ``d_avatar.params``.

The eye corners and eyelid points are chosen from the official eye contour
sets by geometry: corner = extreme x of the contour; lid pair = the contour
points nearest in x to the iBUG lid positions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# FACEMESH vertex sets, fetched from the official MediaPipe source
# mediapipe/python/solutions/face_mesh_connections.py and printed before use.
FACEMESH_LEFT_EYE = frozenset(
    [
        (263, 249),
        (249, 390),
        (390, 373),
        (373, 374),
        (374, 380),
        (380, 381),
        (381, 382),
        (382, 362),
        (263, 466),
        (466, 388),
        (388, 387),
        (387, 386),
        (386, 385),
        (385, 384),
        (384, 398),
        (398, 362),
    ]
)
FACEMESH_RIGHT_EYE = frozenset(
    [
        (33, 7),
        (7, 163),
        (163, 144),
        (144, 145),
        (145, 153),
        (153, 154),
        (154, 155),
        (155, 133),
        (33, 246),
        (246, 161),
        (161, 160),
        (160, 159),
        (159, 158),
        (158, 157),
        (157, 173),
        (173, 133),
    ]
)
FACEMESH_LEFT_EYEBROW = frozenset(
    [
        (276, 283),
        (283, 282),
        (282, 295),
        (295, 285),
        (300, 293),
        (293, 334),
        (334, 296),
        (296, 336),
    ]
)
FACEMESH_RIGHT_EYEBROW = frozenset(
    [
        (46, 53),
        (53, 52),
        (52, 65),
        (65, 55),
        (70, 63),
        (63, 105),
        (105, 66),
        (66, 107),
    ]
)
FACEMESH_LIPS = frozenset(
    [
        (61, 146),
        (146, 91),
        (91, 181),
        (181, 84),
        (84, 17),
        (17, 314),
        (314, 405),
        (405, 321),
        (321, 375),
        (375, 291),
        (61, 185),
        (185, 40),
        (40, 39),
        (39, 37),
        (37, 0),
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
        (78, 95),
        (95, 88),
        (88, 178),
        (178, 87),
        (87, 14),
        (14, 317),
        (317, 402),
        (402, 318),
        (318, 324),
        (324, 308),
        (78, 191),
        (191, 80),
        (80, 81),
        (81, 82),
        (82, 13),
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
    ]
)
FACEMESH_NOSE = frozenset(
    [
        (168, 6),
        (6, 197),
        (197, 195),
        (195, 5),
        (5, 4),
        (4, 1),
        (1, 19),
        (19, 94),
        (94, 2),
        (98, 97),
        (97, 2),
        (2, 326),
        (326, 327),
        (327, 294),
        (294, 278),
        (278, 344),
        (344, 440),
        (440, 275),
        (275, 4),
        (4, 45),
        (45, 220),
        (220, 115),
        (115, 48),
        (48, 64),
        (64, 98),
    ]
)
FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477), (477, 474)])
FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472), (472, 469)])

_MIN_POINTS = 468
_MAX_POINTS = 478
_NDIM = 2
_COORD_COUNTS = (2, 3)
_MOUTH_OUTER_COUNT = 12
_MOUTH_INNER_END = 20
_EAR_OPEN_DEFAULT = 0.3


def _contour_loop(edges: frozenset[tuple[int, int]]) -> list[int]:
    """Order the vertex indices of a closed contour edge set into a loop."""
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = next(iter(adj))
    loop = [start]
    prev: int | None = None
    cur = start
    while True:
        nbs = [n for n in adj[cur] if n != prev]
        if not nbs:
            break
        nxt = nbs[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(adj) + 2:
            break
    return loop


def _arc_between(loop: Sequence[int], i_a: int, i_b: int, forward: bool) -> list[int]:
    """The ordered list of indices from loop[i_a] to loop[i_b] along the loop."""
    n = len(loop)
    arc: list[int] = []
    i = i_a
    step = 1 if forward else -1
    while True:
        arc.append(loop[i])
        if i == i_b:
            break
        i = (i + step) % n
        if len(arc) > n + 2:
            break
    return arc


def _lid_point(
    mesh: np.ndarray,
    arc: Sequence[int],
    corner_min: np.ndarray,
    corner_max: np.ndarray,
    third: float,
) -> int:
    """Index of the contour point on ``arc`` nearest in x to an iBUG lid position.

    iBUG lid positions sit at ``third`` and ``1 - third`` of the x distance
    between the two corners.
    """
    target_x = float(corner_min[0]) + third * float(corner_max[0] - corner_min[0])
    # The arc runs corner_min -> corner_max, so the endpoints are the corners.
    corner_idxs = {int(arc[0]), int(arc[-1])}
    best_idx = int(arc[0])
    best_dist = float('inf')
    for idx in arc:
        if int(idx) in corner_idxs:
            continue
        dist = abs(float(mesh[idx, 0]) - target_x)
        if dist < best_dist:
            best_dist = dist
            best_idx = int(idx)
    return best_idx


def _eye_map(
    mesh: np.ndarray,
    contour: frozenset[tuple[int, int]],
    is_left: bool,
) -> dict[str, int]:
    """Map one eye contour to its six iBUG eye indices (36-41 or 42-47).

    Returns a dict with keys ``outer``, ``inner``, ``top1``, ``top2``,
    ``bot1``, ``bot2`` holding mesh vertex indices.
    """
    loop = _contour_loop(contour)
    loop_pts = mesh[list(loop)]
    xs = loop_pts[:, 0]
    i_min = int(np.argmin(xs))
    i_max = int(np.argmax(xs))
    arc_fwd = _arc_between(loop, i_min, i_max, True)
    arc_bwd = _arc_between(loop, i_min, i_max, False)
    pts_fwd = mesh[arc_fwd]
    pts_bwd = mesh[arc_bwd]
    if pts_fwd[:, 1].mean() <= pts_bwd[:, 1].mean():
        top_arc, bot_arc = arc_fwd, arc_bwd
    else:
        top_arc, bot_arc = arc_bwd, arc_fwd
    corner_min = mesh[int(loop[i_min])]
    corner_max = mesh[int(loop[i_max])]
    if is_left:
        # iBUG left eye: 36 = outer = min-x corner, 39 = inner = max-x corner.
        outer_idx = int(loop[i_min])
        inner_idx = int(loop[i_max])
    else:
        # iBUG right eye: 42 = inner = min-x corner, 45 = outer = max-x corner.
        inner_idx = int(loop[i_min])
        outer_idx = int(loop[i_max])
    top1 = _lid_point(mesh, top_arc, corner_min, corner_max, 1.0 / 3.0)
    top2 = _lid_point(mesh, top_arc, corner_min, corner_max, 2.0 / 3.0)
    bot1 = _lid_point(mesh, bot_arc, corner_min, corner_max, 1.0 / 3.0)
    bot2 = _lid_point(mesh, bot_arc, corner_min, corner_max, 2.0 / 3.0)
    return {
        'outer': outer_idx,
        'inner': inner_idx,
        'top1': top1,
        'top2': top2,
        'bot1': bot1,
        'bot2': bot2,
    }


def mesh_to_68(mesh: np.ndarray) -> np.ndarray:  # noqa: PLR0915
    """Convert one face mesh to a (68, 2) frame in the iBUG ordering.

    Args:
        mesh: (468, 2) or (468, 3) base mesh, or (478, 2)/(478, 3) refined
            mesh that also carries the iris points.

    Returns:
        (68, 2) array of (x, y) coordinates in the iBUG ordering. Jaw points
        0-16 are NaN (no mesh counterpart, never read).

    Raises:
        ValueError: if ``mesh`` is not (468 or 478, 2 or 3).
    """
    pts = np.asarray(mesh, dtype=float)
    if (
        pts.ndim != _NDIM
        or pts.shape[0] not in (_MIN_POINTS, _MAX_POINTS)
        or pts.shape[1] not in _COORD_COUNTS
    ):
        raise ValueError(f'Expected mesh of shape (468 or 478, 2 or 3), got {pts.shape}')
    xy = pts[:, :2]
    out: np.ndarray = np.full((68, 2), np.nan, dtype=float)

    left_map = _eye_map(xy, FACEMESH_LEFT_EYE, is_left=True)
    right_map = _eye_map(xy, FACEMESH_RIGHT_EYE, is_left=False)

    # Assign contours to iBUG eyes by image x position so head_roll stays
    # level for a frontal face: lower mean x -> iBUG 36-41, higher -> 42-47.
    left_pts = xy[list(FACEMESH_LEFT_EYE)]
    right_pts = xy[list(FACEMESH_RIGHT_EYE)]
    if float(left_pts[:, 0].mean()) <= float(right_pts[:, 0].mean()):
        first, second = left_map, right_map
    else:
        first, second = right_map, left_map

    # iBUG 36-41: left eye.
    out[36] = xy[first['outer']]
    out[37] = xy[first['top1']]
    out[38] = xy[first['top2']]
    out[39] = xy[first['inner']]
    out[40] = xy[first['bot1']]
    out[41] = xy[first['bot2']]
    # iBUG 42-47: right eye.
    out[42] = xy[second['inner']]
    out[43] = xy[second['top1']]
    out[44] = xy[second['top2']]
    out[45] = xy[second['outer']]
    out[46] = xy[second['bot1']]
    out[47] = xy[second['bot2']]

    # Brows: 17-26 from the two eyebrow contours (5 points each).
    brow_contours = [FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW]
    for k, brow in enumerate(brow_contours):
        b = np.array(sorted({i for e in brow for i in e}), dtype=int)
        bpts = xy[b]
        order = np.argsort(bpts[:, 0])
        for j, idx in enumerate(order):
            out[17 + k * 5 + j] = xy[int(b[idx])]

    # Nose: 27-35 from the nose contour, tip at the lowest y (image coords).
    nose = np.array(sorted({i for e in FACEMESH_NOSE for i in e}), dtype=int)
    nose_pts = xy[nose]
    tip = nose[int(np.argmax(nose_pts[:, 1]))]
    out[30] = xy[int(tip)]
    # Nose bridge between the eyes: use the two nose points nearest the eye
    # inner corners, spread across indices 27-31.
    left_inner = xy[first['inner']]
    right_inner = xy[second['inner']]
    bridge_x = (float(left_inner[0]) + float(right_inner[0])) / 2.0
    bridge_y = min(float(left_inner[1]), float(right_inner[1])) - 2.0
    out[27:31] = np.stack([np.full(4, bridge_x), np.linspace(bridge_y, bridge_y + 6.0, 4)], axis=1)
    out[31:36] = out[30]

    # Mouth: 48-67 from the lips contour (outer + inner lips).
    lips = np.array(sorted({i for e in FACEMESH_LIPS for i in e}), dtype=int)
    lip_pts = xy[lips]
    cx = float(lip_pts[:, 0].mean())
    cy = float(lip_pts[:, 1].mean())
    order = np.argsort(lip_pts[:, 0])
    outer = lips[order[:_MOUTH_OUTER_COUNT]]
    inner = (
        lips[order[_MOUTH_OUTER_COUNT:_MOUTH_INNER_END]]
        if len(order) >= _MOUTH_INNER_END
        else lips[order[_MOUTH_OUTER_COUNT:]]
    )
    for j, idx in enumerate(outer):
        out[48 + j] = xy[int(idx)]
    for j, idx in enumerate(inner):
        out[60 + j] = xy[int(idx)]
    # Centre the mouth on the lips' mean so a closed mouth reads neutral.
    shift = np.array([cx - out[48:68, 0].mean(), cy - out[48:68, 1].mean()])
    for k in range(48, 68):
        out[k] = out[k] + shift
    return out


def iris_centres(mesh: np.ndarray) -> np.ndarray:
    """Return the left and right iris centres as a (2, 2) array.

    Args:
        mesh: (468, 2)/(468, 3) or (478, 2)/(478, 3) face mesh.

    Returns:
        (2, 2) array ``[left_iris, right_iris]`` in the iBUG eye order
        (left eye = lower mean x contour). For a 468-point mesh (no iris
        vertices) both centres are zero.
    """
    pts = np.asarray(mesh, dtype=float)
    if (
        pts.ndim != _NDIM
        or pts.shape[0] not in (_MIN_POINTS, _MAX_POINTS)
        or pts.shape[1] not in _COORD_COUNTS
    ):
        raise ValueError(f'Expected mesh of shape (468 or 478, 2 or 3), got {pts.shape}')
    xy = pts[:, :2]
    left_pts = xy[list(FACEMESH_LEFT_EYE)]
    right_pts = xy[list(FACEMESH_RIGHT_EYE)]
    if float(left_pts[:, 0].mean()) <= float(right_pts[:, 0].mean()):
        first_iris, second_iris = FACEMESH_RIGHT_IRIS, FACEMESH_LEFT_IRIS
    else:
        first_iris, second_iris = FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS

    def centre(iris_edges: frozenset[tuple[int, int]]) -> np.ndarray:
        idx = np.array(sorted({i for e in iris_edges for i in e}), dtype=int)
        if idx.size == 0 or int(idx.max()) >= xy.shape[0]:
            return np.zeros(2, dtype=float)
        return xy[idx].mean(axis=0).astype(float)  # type: ignore[no-any-return]

    return np.stack([centre(first_iris), centre(second_iris)], axis=0).astype(float)  # type: ignore[no-any-return]
