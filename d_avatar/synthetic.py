"""Synthetic 68-point face, in the dlib/iBUG ordering, used by the examples and the tests."""

from __future__ import annotations

import numpy as np


def synthetic_face_68() -> np.ndarray:
    """Return a neutral face of shape (68, 2).

    Geometry (pixels): eye width 10, eye height 3 (eye aspect ratio 0.3), eyebrows
    15 px above the eye centre, mouth width 20 and lip gap 3, eye centres 40 px apart.
    """
    pts: np.ndarray = np.zeros((68, 2), dtype=np.float64)
    pts[17:22] = [[20.0 + 5.0 * i, 20.0] for i in range(5)]
    pts[22:27] = [[60.0 + 5.0 * i, 20.0] for i in range(5)]
    pts[36:42] = [[20, 35], [23, 33.5], [27, 33.5], [30, 35], [27, 36.5], [23, 36.5]]
    pts[42:48] = [[60, 35], [63, 33.5], [67, 33.5], [70, 35], [67, 36.5], [63, 36.5]]
    pts[27:36] = np.stack([np.full(9, 45.0), np.linspace(38.0, 52.0, 9)], axis=1)
    pts[48:68] = [45.0, 70.0]
    pts[48] = [35.0, 70.0]
    pts[54] = [55.0, 70.0]
    pts[51:53] = [[43.0, 68.5], [47.0, 68.5]]
    pts[57:59] = [[47.0, 71.5], [43.0, 71.5]]
    return pts
