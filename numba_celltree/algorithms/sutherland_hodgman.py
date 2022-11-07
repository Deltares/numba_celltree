"""
Sutherland-Hodgman clipping
---------------------------
Vertices (always lower case, single letter):
Clipping polygon with vertices r, s, ...
Subject polgyon with vertices a, b, ...
Vectors (always upper case, single letter):

* U: r -> s
* N: norm, orthogonal to u
* V: a -> b
* W: a -> r

   s ----- ...
   |
   |  b ----- ...
   | /
   |/
   x
  /|
 / |
a--+-- ...
   |
   r ----- ...

Floating point rounding should not be an issue, since we're only looking at
finding the area of overlap of two convex polygons.
In case of intersection failure, we can ignore it when going out -> in. It will
occur when the outgoing point is very close the clipping edge. In that case the
intersection point ~= vertex b, and we can safely skip the intersection.
When going in -> out, b might be located on the edge. If intersection fails,
again the intersection point ~= vertex b. We treat b as if it is just on the
inside and append it. For consistency, we set b_inside to True, as it will be
used as a_inside in the next iteration.
"""
from typing import Sequence, Tuple

import numba as nb
import numpy as np

from ..constants import PARALLEL, FloatArray, FloatDType, IntArray
from ..geometry_utils import (
    Point,
    Vector,
    as_box,
    as_point,
    copy_box_vertices,
    copy_vertices,
    dot_product,
    polygon_area,
)
from ..utils import allocate_clip_polygon, copy


@nb.njit(inline="always")
def inside(p: Point, r: Point, U: Vector):
    # U: a -> b direction vector
    # p is point r or s
    return U.x * (p.y - r.y) > U.y * (p.x - r.x)


@nb.njit(inline="always")
def intersection(a: Point, V: Vector, r: Point, N: Vector) -> Tuple[bool, Point]:
    # Find the intersection with an (infinite) clipping plane
    W = Vector(r.x - a.x, r.y - a.y)
    nw = dot_product(N, W)
    nv = dot_product(N, V)
    if nv != 0:
        t = nw / nv
        return True, Point(a.x + t * V.x, a.y + t * V.y)
    else:
        # parallel lines
        return False, Point(np.nan, np.nan)


@nb.njit(inline="always")
def push_point(polygon: FloatArray, size: int, p: Point) -> int:
    polygon[size][0] = p.x
    polygon[size][1] = p.y
    return size + 1


@nb.njit(inline="always")
def polygon_polygon_clip_area(polygon: Sequence, clipper: Sequence) -> float:
    n_output = len(polygon)
    n_clip = len(clipper)
    subject = allocate_clip_polygon()
    output = allocate_clip_polygon()

    # Copy polygon into output
    copy(polygon, output, n_output)

    # Grab last point
    r = as_point(clipper[n_clip - 1])
    for i in range(n_clip):
        s = as_point(clipper[i])

        U = Vector(s.x - r.x, s.y - r.y)
        if U.x == 0 and U.y == 0:
            continue
        N = Vector(-U.y, U.x)

        # Copy output into subject
        length = n_output
        copy(output, subject, length)
        # Reset
        n_output = 0
        # Grab last point
        a = as_point(subject[length - 1])
        a_inside = inside(a, r, U)
        for j in range(length):
            b = as_point(subject[j])

            V = Vector(b.x - a.x, b.y - a.y)
            if V.x == 0 and V.y == 0:
                continue

            b_inside = inside(b, r, U)
            if b_inside:
                if not a_inside:  # out, or on the edge
                    succes, point = intersection(a, V, r, N)
                    if succes:
                        n_output = push_point(output, n_output, point)
                n_output = push_point(output, n_output, b)
            elif a_inside:
                succes, point = intersection(a, V, r, N)
                if succes:
                    n_output = push_point(output, n_output, point)
                else:  # Floating point failure
                    # TODO: haven't come up with a test case yet to succesfully
                    # trigger this ...
                    b_inside = True  # flip it for consistency, will be set as a
                    n_output = push_point(output, n_output, b)  # push b instead

            # Advance to next polygon edge
            a = b
            a_inside = b_inside

        # Exit early in case not enough vertices are left.
        if n_output < 3:
            return 0.0

        # Advance to next clipping edge
        r = s

    area = polygon_area(output[:n_output])
    return area


@nb.njit(parallel=PARALLEL, cache=True)
def area_of_intersection(
    vertices_a: FloatArray,
    vertices_b: FloatArray,
    faces_a: IntArray,
    faces_b: IntArray,
    indices_a: IntArray,
    indices_b: IntArray,
) -> FloatArray:
    n_intersection = indices_a.size
    area = np.empty(n_intersection, dtype=FloatDType)
    for i in nb.prange(n_intersection):
        face_a = faces_a[indices_a[i]]
        face_b = faces_b[indices_b[i]]
        a = copy_vertices(vertices_a, face_a)
        b = copy_vertices(vertices_b, face_b)
        area[i] = polygon_polygon_clip_area(a, b)
    return area


@nb.njit(parallel=PARALLEL, cache=True)
def box_area_of_intersection(
    bbox_coords: FloatArray,
    vertices: FloatArray,
    faces: IntArray,
    indices_bbox: IntArray,
    indices_face: IntArray,
) -> FloatArray:
    n_intersection = indices_bbox.size
    area = np.empty(n_intersection, dtype=FloatDType)
    for i in nb.prange(n_intersection):
        box = as_box(bbox_coords[indices_bbox[i]])
        face = faces[indices_face[i]]
        a = copy_box_vertices(box)
        b = copy_vertices(vertices, face)
        area[i] = polygon_polygon_clip_area(a, b)
    return area
