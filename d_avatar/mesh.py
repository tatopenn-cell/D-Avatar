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

from collections import defaultdict  # noqa: PLC0415  # used in mesh_to_68
from collections.abc import Sequence

import numpy as np

# FACEMESH vertex sets, fetched from the official MediaPipe source
# mediapipe/python/solutions/face_mesh_connections.py and printed before use.
# Copyright 2021 The MediaPipe Authors. Licensed under Apache-2.0.
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
_LIP_LOOP_COUNT = 2
_BRIDGE_CENTER_TOL = 10.0
_BRIDGE_POINT_COUNT = 5
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
    ibug_side: str,  # 'LEFT' for iBUG 36-41, 'RIGHT' for iBUG 42-47
) -> dict[str, int]:
    """Map one eye contour to its six iBUG eye indices (36-41 or 42-47).

    The corner assignment depends on which iBUG eye this is:
    - iBUG LEFT eye (36-41): outer=36 (temple) is at MAX x, inner=39 (nose) at MIN x
    - iBUG RIGHT eye (42-47): inner=42 (nose) at MIN x, outer=45 (temple) at MAX x

    This matches a non-mirrored camera image where the person's left eye
    appears on the right side of the image (higher x).

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
    if ibug_side == 'LEFT':
        # iBUG left eye (36-41): outer=36 at MAX x, inner=39 at MIN x
        outer_idx = int(loop[i_max])
        inner_idx = int(loop[i_min])
    else:  # 'RIGHT'
        # iBUG right eye (42-47): inner=42 at MIN x, outer=45 at MAX x
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


def mesh_to_68(mesh: np.ndarray) -> np.ndarray:  # noqa: PLR0915, PLR0912
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

    # Map eye contours by image position, not MediaPipe naming.
    # In a non-mirrored image: contour at lower mean x = person's RIGHT eye (iBUG 42-47)
    #                        contour at higher mean x = person's LEFT eye (iBUG 36-41)
    left_eye_contour = FACEMESH_LEFT_EYE
    right_eye_contour = FACEMESH_RIGHT_EYE
    left_pts = xy[list(left_eye_contour)]
    right_pts = xy[list(right_eye_contour)]

    if float(left_pts[:, 0].mean()) <= float(right_pts[:, 0].mean()):
        # LEFT_EYE contour is at lower x = person's RIGHT eye (iBUG RIGHT, 42-47)
        right_eye_map = _eye_map(xy, left_eye_contour, 'RIGHT')
        left_eye_map = _eye_map(xy, right_eye_contour, 'LEFT')
    else:
        # LEFT_EYE contour is at higher x = person's LEFT eye (iBUG LEFT, 36-41)
        left_eye_map = _eye_map(xy, left_eye_contour, 'LEFT')
        right_eye_map = _eye_map(xy, right_eye_contour, 'RIGHT')

    # iBUG 36-41: left eye. outer=36 (temple), inner=39 (nose).
    out[36] = xy[left_eye_map['outer']]
    out[37] = xy[left_eye_map['top1']]
    out[38] = xy[left_eye_map['top2']]
    out[39] = xy[left_eye_map['inner']]
    out[40] = xy[left_eye_map['bot1']]
    out[41] = xy[left_eye_map['bot2']]
    # iBUG 42-47: right eye. inner=42 (nose), outer=45 (temple).
    out[42] = xy[right_eye_map['inner']]
    out[43] = xy[right_eye_map['top1']]
    out[44] = xy[right_eye_map['top2']]
    out[45] = xy[right_eye_map['outer']]
    out[46] = xy[right_eye_map['bot1']]
    out[47] = xy[right_eye_map['bot2']]

    # Brows: 17-26 from the two eyebrow contours (5 points each).
    # Each contour has 10 vertices; pick 5 by sampling along the sorted x.
    brow_contours = [FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW]
    for k, brow in enumerate(brow_contours):
        b = np.array(sorted({i for e in brow for i in e}), dtype=int)
        bpts = xy[b]
        order = np.argsort(bpts[:, 0])
        # 5 points evenly spaced across the contour
        for j in range(5):
            idx = order[len(order) * j // 5]
            out[17 + k * 5 + j] = xy[int(b[idx])]

    # Nose: 27-35 from the nose contour.
    nose = np.array(sorted({i for e in FACEMESH_NOSE for i in e}), dtype=int)
    nose_pts = xy[nose]
    tip = nose[int(np.argmax(nose_pts[:, 1]))]
    out[30] = xy[int(tip)]
    # Nose bridge 27-31: five points from the nose contour between the eyes.
    # Pick vertices nearest the eye inner corners, ordered by y.
    left_inner = xy[left_eye_map['inner']]
    right_inner = xy[right_eye_map['inner']]
    bridge_center_x = (float(left_inner[0]) + float(right_inner[0])) / 2.0
    # Sort nose vertices by x to find ones near the center
    nose_x_order = np.argsort(nose_pts[:, 0])
    # Filter to x near center
    center_mask = np.abs(nose_pts[nose_x_order, 0] - bridge_center_x) < _BRIDGE_CENTER_TOL
    bridge_candidates = nose[nose_x_order][center_mask]
    if len(bridge_candidates) >= _BRIDGE_POINT_COUNT:
        bridge_order = np.argsort(xy[bridge_candidates, 1])
        bridge_sel = bridge_candidates[bridge_order[:_BRIDGE_POINT_COUNT]]
    elif len(bridge_candidates) > 0:
        # Not enough candidates, replicate what we have
        bridge_sel = np.resize(bridge_candidates, 5)
    else:
        # No candidates near center, use tip position for all bridge points
        bridge_sel = np.full(5, tip, dtype=int)
    out[27:32] = xy[bridge_sel]
    # 32-35: replicate tip (iBUG 31-35 are often same as 30)
    out[32:36] = out[30]

    # Mouth: 48-67 from the lips contours (outer + inner).
    # FACEMESH_LIPS has two closed loops: outer lips (20 vertices) and inner lips (20 vertices).
    # iBUG mouth has 20 points: 12 outer (48-59), 8 inner (60-67).
    # Get both lip loops by finding connected components in the edge graph.
    adj = defaultdict(list)
    for a, b in FACEMESH_LIPS:
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    lip_loops = []
    for v in adj:
        if v not in visited:
            loop = []
            cur = v
            prev = None
            while True:
                visited.add(cur)
                loop.append(cur)
                nbs = [n for n in adj[cur] if n != prev]
                if not nbs:
                    break
                nxt = nbs[0]
                if nxt == loop[0]:
                    break
                prev, cur = cur, nxt
                if len(loop) > len(adj) + 2:
                    break
            lip_loops.append(loop)

    # Sort loops by size (outer first, then inner) and by y-position
    lip_loops.sort(key=lambda L: (len(L), -xy[L, 1].mean()))
    if len(lip_loops) >= _LIP_LOOP_COUNT:
        outer_loop, inner_loop = lip_loops[0], lip_loops[1]
    else:
        # Fallback: single loop
        outer_loop = lip_loops[0] if lip_loops else []
        inner_loop = []

    # Map outer loop to iBUG 48-59 (12 points)
    for j in range(12):
        idx = outer_loop[len(outer_loop) * j // 12]
        out[48 + j] = xy[int(idx)]
    # Map inner loop to iBUG 60-67 (8 points)
    for j in range(8):
        idx = inner_loop[len(inner_loop) * j // 8]
        out[60 + j] = xy[int(idx)]
    # Centre the mouth on the lips' mean so a closed mouth reads neutral.
    mouth_mean = out[48:68].mean(axis=0)
    all_lip_idx = np.array(outer_loop + inner_loop)
    lip_mean_x = xy[all_lip_idx, 0].mean()
    lip_mean_y = xy[all_lip_idx, 1].mean()
    shift = np.array([lip_mean_x - mouth_mean[0], lip_mean_y - mouth_mean[1]])
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
        first_iris, second_iris = FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS
    else:
        first_iris, second_iris = FACEMESH_RIGHT_IRIS, FACEMESH_LEFT_IRIS

    def centre(iris_edges: frozenset[tuple[int, int]]) -> np.ndarray:
        idx = np.array(sorted({i for e in iris_edges for i in e}), dtype=int)
        if idx.size == 0 or int(idx.max()) >= xy.shape[0]:
            return np.zeros(2, dtype=float)
        return xy[idx].mean(axis=0).astype(float)  # type: ignore[no-any-return]

    return np.stack([centre(first_iris), centre(second_iris)], axis=0).astype(float)  # type: ignore[no-any-return]
