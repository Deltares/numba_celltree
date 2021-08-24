from typing import Sequence, Tuple

import numba as nb
import numpy as np

from .constants import FLOAT_MAX, FLOAT_MIN, PARALLEL, BoolArray, FloatArray, IntArray
from .geometry_utils import Point, Vector, copy_vertices, dot_product


@nb.njit(inline="always")
def extrema_projected(
    norm: Vector, polygon: Sequence[Point], length: int
) -> Tuple[float, float]:
    min_proj = FLOAT_MAX
    max_proj = FLOAT_MIN
    for i in range(length):
        proj = dot_product(Point(polygon[i][0], polygon[i][1]), norm)
        min_proj = min(min_proj, proj)
        max_proj = max(max_proj, proj)
    return min_proj, max_proj


@nb.njit(inline="always")
def is_separating_axis(
    norm: Vector, a: Sequence[Point], b: Sequence[Point], length_a: int, length_b: int
) -> bool:
    mina, maxa = extrema_projected(norm, a, length_a)
    minb, maxb = extrema_projected(norm, b, length_b)
    if maxa > minb and maxb > mina:
        return False
    else:
        return True


@nb.njit(inline="always")
def separating_axes(
    a: Sequence[Point], b: Sequence[Point], length_a: int, length_b: int
) -> bool:
    p = Point(a[length_a - 1][0], a[length_a - 1][1])
    for i in range(length_a):
        q = Point(a[i][0], a[i][1])
        norm = Vector(p.y - q.y, q.x - p.x)
        if norm.x == 0.0 and norm.y == 0.0:
            continue
        if is_separating_axis(norm, a, b, length_a, length_b):
            return False
        p = q
    return True


@nb.njit(parallel=PARALLEL)
def polygons_intersect(
    vertices_a: FloatArray,
    vertices_b: FloatArray,
    faces_a: IntArray,
    faces_b: IntArray,
    indices_a: IntArray,
    indices_b: IntArray,
) -> BoolArray:
    n_shortlist = indices_a.size
    intersects = np.empty(n_shortlist, dtype=np.bool)
    for i in nb.prange(n_shortlist):
        face_a = faces_a[indices_a[i]]
        face_b = faces_b[indices_b[i]]
        a, length_a = copy_vertices(vertices_a, face_a)
        b, length_b = copy_vertices(vertices_b, face_b)
        intersects[i] = separating_axes(a, b, length_a, length_b) and separating_axes(
            b, a, length_b, length_a
        )
