"""
Implementation based off the description in:
Skala, V. (1993). An efficient algorithm for line clipping by convex polygon.
Computers & Graphics, 17(4), 417-421.

Available at:
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.5729&rep=rep1&type=pdf

Also available in:
Duc Huy Bui, 1999. Algorithms for Line Clipping and Their Complexity. PhD
Thesis.

Available at:
http://graphics.zcu.cz/files/DIS_1999_Bui_Duc_Huy.pdf
"""

from typing import Sequence, Tuple

import numba as nb
import numpy as np

from numba_celltree.constants import TOLERANCE_ON_EDGE, Point, Vector
from numba_celltree.geometry_utils import (
    as_point,
    cross_product,
    dot_product,
    point_in_polygon_or_on_edge,
    to_point,
    to_vector,
)

NO_INTERSECTION = False, Point(np.nan, np.nan), Point(np.nan, np.nan)


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
    v0 = as_point(poly[i0])
    v01 = as_point(poly[(i0 + 1) % length])
    v1 = as_point(poly[i1])
    v11 = as_point(poly[(i1 + 1) % length])
    _, t0 = compute_intersection(a, s, v0, v01)
    enters1, t1 = compute_intersection(a, s, v1, v11)
    if enters1:  # Swap them
        return t1, t0
    else:
        return t0, t1


@nb.njit(inline="always")
def overlap(ta: Point, tb: Point, t0: Point, t1: Point) -> bool:
    if ta > tb:
        ta, tb = tb, ta
    if t0 > t1:
        t0, t1 = t1, t0
    vector_overlap = max(0, min(tb, t1) - max(ta, t0))
    return vector_overlap > 0.0


@nb.njit(inline="always")
def aligned(U: Vector, V: Vector) -> bool:
    # Any zero vector: always aligned.
    if (U.x == 0 and U.y == 0) or (V.x == 0 and V.y == 0):
        return True

    # Both x-components non-zero:
    if U.x != 0 and V.x != 0:
        return (U.x > 0) == (V.x > 0)

    # Both y-components non-zero:
    if U.y != 0 and V.y != 0:
        return (U.y > 0) == (V.y > 0)

    # One vertical, one horizontal: not aligned.
    return False


@nb.njit(inline="always")
def collinear_case(a: Point, b: Point, v0: Point, v1: Point) -> Tuple[Point, Point]:
    # Redefine everything relative to point a to avoid precision loss in cross
    # products.
    # _a is implicit (0.0, 0.0)
    _b = Point(b.x - a.x, b.y - a.y)
    _v0 = Point(v0.x - a.x, v0.y - a.y)
    _v1 = Point(v1.x - a.x, v1.y - a.y)

    # Check orientation
    U = Vector(_b.x, _b.y)
    V = to_vector(_v0, _v1)
    if not aligned(U, V):
        v0, v1 = v1, v0
        _v0, _v1 = _v1, _v0

    # Project on the same axis (t), take inner values
    n = Vector(-_b.y, _b.x)  # a implicit
    ta = 0.0  # a implicit
    tb = cross_product(n, _b)
    t0 = cross_product(n, _v0)
    t1 = cross_product(n, _v1)

    if not overlap(ta, tb, t0, t1):
        return NO_INTERSECTION

    if t0 < ta:
        p0 = v0
    else:
        p0 = a

    if t1 > tb:
        p1 = v1
    else:
        p1 = b

    return True, p0, p1


# Too big to inline. Drives compilation time through the roof for no benefit.
@nb.njit(inline="never")
def cyrus_beck_line_polygon_clip(
    a: Point, b: Point, poly: Sequence[Point]
) -> Tuple[bool, Point, Point]:
    """
    In short, the basic idea:

    For a given segment s (a -> b), test which two edges of a convex polygon it
    can intersect. If it intersects, the vertices [v0, v1] of an edge are
    separated by the segment (s). If s separates, the cross products of a -> v0
    (ksi) and a -> v1 (eta) will point in opposing directions (ksi * eta < 0).
    If both are > 0 or both are < 0 (ksi * eta > 0), they fall on the on the
    same side of the line; they are parallel and possibly collinear if ksi *
    eta == 0.

    Once the number of intersections (k), and the possibly intersecting edges
    (i0, i1) have been identified, we can compute the intersections. This
    assumes the vertices of the polygons are ordered in counter-clockwise
    orientation. We can also tell whether a line is possibly entering or
    leaving the polygon by the sign of the dot product.

    A valid intersection falls on the domain of the parametrized segment:
    0 <= t <= 1.0
    """
    tolerance = TOLERANCE_ON_EDGE
    length = len(poly)
    s = to_vector(a, b)

    # Test whether points are identical
    if s.x == 0 and s.y == 0:
        return NO_INTERSECTION
    # Test whether line is fully enclosed in polygon
    a_inside = point_in_polygon_or_on_edge(a, poly, tolerance)
    b_inside = point_in_polygon_or_on_edge(b, poly, tolerance)
    if a_inside and b_inside:
        return True, a, b

    i0 = -1
    i1 = -1
    i = 0
    k = 0
    v = as_point(poly[i])
    ksi = cross_product(to_vector(a, v), s)

    while i < length and k < 2:
        v0 = as_point(poly[i])
        v1 = as_point(poly[(i + 1) % length])
        # Check if they can cross at all
        eta = cross_product(to_vector(a, v1), s)

        # Note; ksi * eta < 0 doesn't work as well
        if (ksi < 0.0) ^ (eta < 0.0):
            if k == 0:
                i0 = i
            else:
                i1 = i
            k += 1
        # Calculate the area of the triangle formed by a, b, v0
        # if zero, then points are collinear.
        elif (ksi == 0.0) and (eta == 0.0):
            return collinear_case(a, b, v0, v1)

        # Don't recompute ksi
        ksi = eta
        i += 1

    if k == 0:
        return NO_INTERSECTION

    # Gather the intersections, given half-planes
    t0, t1 = intersections(a, s, poly, length, k, i0, i1)

    # Deal with edge cases
    if t0 == t1:
        if a_inside and t1 != 0.0:
            t0 = 0.0
        elif b_inside and t0 != 1.0:
            t1 = 1.0
        else:
            return NO_INTERSECTION

    # Swap if necessary so that t0 is the smaller
    if t1 < t0:
        t0, t1 = t1, t0

    # Note:
    # t >= 0, not >
    # t1 <= 1, not <
    valid0 = t0 >= 0 and t0 < 1
    valid1 = t1 > 0 and t1 <= 1

    # Return only the intersections that are within the segment
    if valid0 and valid1:
        return True, to_point(t0, a, s), to_point(t1, a, s)
    elif valid0:
        return True, to_point(t0, a, s), b
    elif valid1:
        return True, a, to_point(t1, a, s)
    else:
        return NO_INTERSECTION
