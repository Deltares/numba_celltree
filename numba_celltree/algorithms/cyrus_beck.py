from typing import Sequence, Tuple

import numba as nb
import numpy as np

from ..constants import Point, Vector
from ..geometry_utils import cross_product, dot_product, to_point, to_vector

DUMMY = Point(np.nan, np.nan)
NONE = 0
ENTER = 1
LEAVE = 2
ENTER_LEAVE = 3


@nb.njit(inline="always")
def compute_intersection(
    a: Point, s: Vector, v0: Point, v1: Point
) -> Tuple[bool, float]:
    # Computes intersection for parametrized vector
    # vector given by a & s
    # line given by polygon vertices v0 & v1
    # Due to ksi_eta check, k should never be 0 (parallel, possibly collinear)
    # Note: Polygon must be counter-clockwise
    si = to_vector(a, v0)
    n = Vector(-(v1.y - v0.y), (v1.x - v0.x))
    n_si = dot_product(n, si)
    k = dot_product(n, s)
    t = n_si / k
    if n_si > 0:
        return True, t  # Entering
    else:
        return False, t  # leaving


@nb.njit(inline="always")
def intersections(
    a: Point, s: Vector, poly: Sequence[Point], length: int, k: int, i0: int, i1: int
) -> Tuple[float, float]:
    # Return t's for parametrized vector
    # Note: polygon must be counter-clockwise

    # A single intersection found, could be entering, could be leaving
    if k == 1:
        enters0, t = compute_intersection(a, s, poly[i0], poly[(i0 + 1) % length])
        if enters0:
            return t, 1
        else:
            return 0, t
    # Entering and leaving intersection found
    else:  # k == 2:  # Entering and leaving
        enters0, t0 = compute_intersection(a, s, poly[i0], poly[(i0 + 1) % length])
        enters1, t1 = compute_intersection(a, s, poly[i1], poly[(i1 + 1) % length])
        if enters1:  # Swap them
            return t1, t0
        else:
            return t0, t1


@nb.njit(inline="always")
def is_collinear(a: Point, b: Point, c: Point) -> bool:
    # Calculate the area of the triangle formed by a, b, c
    # if zero, then points are collinear.
    u = to_vector(a, b)
    v = to_vector(c, a)
    # TODO: should be allclose I guess?
    return cross_product(u, v) == 0


@nb.njit(inline="always")
def collinear_case(a: Point, b: Point, v0: Point, v1: Point) -> Tuple[Point, Point]:
    # Project on the same axis (t), take inner values
    n = Vector(-(b.y - a.y), (b.x - a.x))
    ta = cross_product(n, a)
    tb = cross_product(n, b)
    t0 = cross_product(n, v0)
    t1 = cross_product(n, v1)

    if t0 < ta:
        p0 = v0
    else:
        p0 = a

    if t1 > tb:
        p1 = v1
    else:
        p1 = b

    return p0, p1


# @nb.njit
def cyrus_beck_line_clip(
    poly: Sequence[Point], a: Point, b: Point
) -> Tuple[int, Point, Point]:
    # Implementation based off the description in:
    # Skala, V. (1993). An efficient algorithm for line clipping by convex
    # polygon. Computers & Graphics, 17(4), 417-421.
    # Available at:
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.5729&rep=rep1&type=pdf
    #
    # Also available in:
    # Duc Huy Bui, 1999. Algorithms for Line Clipping and Their Complexity. PhD
    # Thesis. Available at:
    # http://graphics.zcu.cz/files/DIS_1999_Bui_Duc_Huy.pdf
    #
    # In short, the basic idea:
    # For a given segment s (a -> b), test which two edges of a convex polygon
    # it can intersect. If it intersects, the vertices [v0, v1] of an edge are
    # separated by the segment (s). If s separates, the cross products of a ->
    # v0 (ksi) and a -> v1 (eta) will point in opposing directions (ksi * eta <
    # 0). If both are > 0 or both are < 0 (ksi * eta > 0), they fall on the on
    # the same side of the line; they are parallel and possibly collinear if ksi
    # * eta == 0.
    #
    # Once the number of intersections (k), and the possibly intersecting edges
    # (i0, i1) have been identified, we can compute the intersections. This
    # assumes the vertices of the polygons are ordered in counter-clockwise
    # orientation. We can also tell whether a line is possibly entering or
    # leaving the polygon by the sign of the dot product.
    #
    # A valid intersection falls on the domain of the parametrized segment:
    # 0 <= t <= 1.
    NO_INTERSECTION = False, Point(np.nan, np.nan), Point(np.nan, np.nan)

    length = len(poly)
    s = to_vector(a, b)

    i0 = -1
    i1 = -1
    i = 0
    k = 0
    ksi = cross_product(to_vector(a, poly[i]), s)
    while i < length and k < 2:
        v0 = poly[i]
        v1 = poly[(i + 1) % length]
        # Check if they can cross at all
        eta = cross_product(to_vector(a, v1), s)
        ksi_eta = ksi * eta
        # Don't recompute ksi
        ksi = eta

        # TODO: Allclose?
        if ksi_eta == 0 and is_collinear(a, v0, v1):
            p0, p1 = collinear_case(a, b, v0, v1)
            return ENTER_LEAVE, p0, p1
        elif (ksi_eta) < 0:
            if k == 0:
                i0 = i
            else:
                i1 = i
            k += 1

        i += 1

    if k == 0:
        return NO_INTERSECTION

    # Gather the intersections, given half-planes
    t0, t1 = intersections(a, s, poly, length, k, i0, i1)
    valid0 = t0 > 0 and t0 < 1
    valid1 = t1 > 0 and t1 < 1

    # Return only the intersections that are within the segment
    if valid0 and valid1:
        return True, to_point(t0, a, s), to_point(t1, a, s)
    elif valid0:
        return True, to_point(t0, a, s), b
    elif valid1:
        return True, a, to_point(t1, a, s)
    else:
        return NO_INTERSECTION
