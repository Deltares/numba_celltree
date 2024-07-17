from typing import Tuple

import numba as nb
import numpy as np

from numba_celltree.constants import Box, Point
from numba_celltree.geometry_utils import point_inside_box


@nb.njit
def liang_barsky_line_box_clip(
    a: Point, b: Point, box: Box
) -> Tuple[bool, Point, Point]:
    NO_INTERSECTION = False, Point(np.nan, np.nan), Point(np.nan, np.nan)
    dx = b.x - a.x
    dy = b.y - a.y

    if dx == 0.0 and dy == 0.0:
        return NO_INTERSECTION
    # Test whether line is fully enclosed in box
    if point_inside_box(a, box) and point_inside_box(b, box):
        return True, a, b

    t0 = 0.0
    t1 = 1.0
    P = (-dx, dx, -dy, dy)
    Q = (
        a.x - box.xmin,
        box.xmax - a.x,
        a.y - box.ymin,
        box.ymax - a.y,
    )

    for p_i, q_i in zip(P, Q):
        if p_i == 0:
            # Test whether line is parallel to box:
            # 1. no x-component (dx == 0), to the left or right
            # 2. no y-component (dy == 0), above or below
            if q_i < 0:
                return NO_INTERSECTION
        else:
            # Compute location on vector (t)
            # Compare against full length (0.0 -> 1.0)
            # or earlier computed values
            t = q_i / p_i
            if p_i < 0:
                if t > t1:
                    return NO_INTERSECTION
                elif t > t0:
                    t0 = t
            elif p_i > 0:
                if t < t0:
                    return NO_INTERSECTION
                elif t < t1:
                    t1 = t

    c = Point(a.x + t0 * dx, a.y + t0 * dy)
    d = Point(a.x + t1 * dx, a.y + t1 * dy)
    return True, c, d
