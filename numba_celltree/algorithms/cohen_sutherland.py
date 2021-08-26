"""
Slightly adapted from:
https://github.com/scivision/lineclipping-python-fortran

(MIT License)
"""
from typing import Tuple

import numba as nb
import numpy as np

from ..constants import Box, Point

INSIDE, LEFT, RIGHT, LOWER, UPPER = 0, 1, 2, 4, 8


@nb.njit(inline="always")
def get_clip(a: Point, box: Box):
    p = INSIDE  # default is inside

    # consider x
    if a.x < box.xmin:
        p |= LEFT
    elif a.x > box.xmax:
        p |= RIGHT

    # consider y
    if a.y < box.ymin:
        p |= LOWER  # bitwise OR
    elif a.y > box.ymax:
        p |= UPPER  # bitwise OR
    return p


@nb.njit(inline="always")
def cohen_sutherland_line_box_clip(a: Point, b: Point, box: Box) -> Tuple[Point, Point]:
    """
    Clips a line to a rectangular area.

    This implements the Cohen-Sutherland line clipping algorithm.  xmin,
    ymax, xmax and ymin denote the clipping area, into which the line
    defined by a.x, a.y (start point) and b.x, b.y (end point) will be
    clipped.

    If the line does not intersect with the rectangular clipping area,
    four None values will be returned as tuple. Otherwise a tuple of the
    clipped line points will be returned in the form (cx1, ca.y, cb.x, cb.y).
    """
    NO_INTERSECTION = False, Point(np.nan, np.nan), Point(np.nan, np.nan)
    dx = b.x - a.x
    dy = b.y - a.y
    if dx == 0.0 and dy == 0.0:
        return NO_INTERSECTION

    # check for trivially outside lines
    k1 = get_clip(a, box)
    k2 = get_clip(b, box)

    # examine non-trivially outside points
    # bitwise OR |
    while (k1 | k2) != 0:
        # if both points are inside box (0000), ACCEPT trivial whole line in
        # box

        # if line trivially outside window, REJECT
        if (k1 & k2) != 0:  # bitwise AND &
            return NO_INTERSECTION

        # non-trivial case, at least one point outside window
        # this is not a bitwise or, it's the word "or"
        opt = k1 or k2  # take first non-zero point, short circuit logic
        if opt & UPPER:  # these are bitwise ANDS
            x = a.x + dx * (box.ymax - a.y) / dy
            y = box.ymax
        elif opt & LOWER:
            x = a.x + dx * (box.ymin - a.y) / dy
            y = box.ymin
        elif opt & RIGHT:
            y = a.y + dy * (box.xmax - a.x) / dx
            x = box.xmax
        elif opt & LEFT:
            y = a.y + dy * (box.xmin - a.x) / dx
            x = box.xmin
        else:
            raise RuntimeError("Undefined clipping state")

        if opt == k1:
            a = Point(x, y)
            k1 = get_clip(a, box)
        elif opt == k2:
            b = Point(x, y)
            k2 = get_clip(b, box)

    return True, a, b
