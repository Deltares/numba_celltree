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
from ..geometry_utils import Point, Vector, copy_vertices, dot_product, polygon_area
from ..utils import allocate_clip_polygon, copy, push


@nb.njit(inline="always")
def inside(p: Point, r: Point, U: Vector):
    # U: a -> b direction vector
    # p is point r or s
    return U.x * (p.y - r.y) > U.y * (p.x - r.x)


@nb.njit(inline="always")
def intersection(a: Point, V: Vector, r: Point, N: Vector) -> Tuple[bool, Point]:
    W = Vector(r.x - a.x, r.y - a.y)
    nw = dot_product(N, W)
    nv = dot_product(N, V)
    if nv != 0:
        t = nw / nv
        return True, Point(a.x + t * V.x, a.y + t * V.y)
    else:
        return False, Point(np.nan, np.nan)


@nb.njit(inline="always")
def clip_polygons(polygon: Sequence, clipper: Sequence) -> float:
    n_output = len(polygon)
    n_clip = len(clipper)
    subject = allocate_clip_polygon()
    output = allocate_clip_polygon()

    # Copy polygon into output
    copy(polygon, output, n_output)

    # Grab last point
    r = Point(clipper[n_clip - 1][0], clipper[n_clip - 1][1])
    for i in range(n_clip):
        s = Point(clipper[i][0], clipper[i][1])

        U = Vector(s.x - r.x, s.y - r.y)
        N = Vector(-U.y, U.x)
        if U.x == 0 and U.y == 0:
            continue

        # Copy output into subject
        length = n_output
        copy(output, subject, length)
        # Reset
        n_output = 0
        # Grab last point
        a = Point(subject[length - 1][0], subject[length - 1][1])
        a_inside = inside(a, r, U)
        for j in range(length):
            b = Point(subject[j][0], subject[j][1])

            V = Vector(b.x - a.x, b.y - a.y)
            if V.x == 0 and V.y == 0:
                continue

            b_inside = inside(b, r, U)
            if b_inside:
                if not a_inside:  # out, or on the edge
                    succes, point = intersection(a, V, r, N)
                    if succes:
                        n_output = push(output, n_output, point)
                n_output = push(output, n_output, b)
            elif a_inside:
                succes, point = intersection(a, V, r, N)
                if succes:
                    n_output = push(output, n_output, point)
                else:  # Floating point failure
                    b_inside = True  # flip it for consistency, will be set as a
                    n_output = push(output, n_output, b)  # push b instead

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


@nb.njit(parallel=PARALLEL)
def area_of_intersection(
    vertices_a: FloatArray,
    vertices_b: FloatArray,
    faces_a: IntArray,
    faces_b: IntArray,
    indices_a: IntArray,
    indices_b: IntArray,
) -> FloatArray:
    n_intersection = indices_a.size
    area = np.empyt(n_intersection, dtype=FloatDType)
    for i in nb.prange(n_intersection):
        face_a = faces_a[indices_a]
        face_b = faces_b[indices_b]
        a = copy_vertices(vertices_a, face_a)
        b = copy_vertices(vertices_b, face_b)
        area[i] = clip_polygons(a, b)
    return area
