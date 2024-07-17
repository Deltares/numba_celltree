"""
This is a straightforward implementation of barycentric interpolation for
triangles.

For barycentric interpolation on N-sided convex polygons, see
barycentric_wachspress.py.
"""

import numba as nb
import numpy as np

from numba_celltree.constants import (
    PARALLEL,
    FloatArray,
    FloatDType,
    IntArray,
    Triangle,
)
from numba_celltree.geometry_utils import (
    Point,
    as_point,
    as_triangle,
    cross_product,
    to_vector,
)


@nb.njit(inline="always")
def compute_weights(triangle: Triangle, p: Point, weights: FloatArray):
    ab = to_vector(triangle.a, triangle.b)
    ac = to_vector(triangle.a, triangle.c)
    ap = to_vector(triangle.a, p)
    Aa = abs(cross_product(ab, ap))
    Ac = abs(cross_product(ac, ap))
    A = abs(cross_product(ab, ac))
    inv_denom = 1.0 / A
    w = inv_denom * Aa
    v = inv_denom * Ac
    u = 1.0 - v - w
    weights[0] = u
    weights[1] = v
    weights[2] = w
    return


@nb.njit(parallel=PARALLEL, cache=True)
def barycentric_triangle_weights(
    points: FloatArray,
    face_indices: IntArray,
    faces: IntArray,
    vertices: FloatArray,
) -> FloatArray:
    n_points = len(points)
    weights = np.zeros((n_points, 3), dtype=FloatDType)
    for i in nb.prange(n_points):
        face_index = face_indices[i]
        if face_index == -1:
            continue
        face = faces[face_index]
        triangle = as_triangle(vertices, face)
        point = as_point(points[i])
        compute_weights(triangle, point, weights[i])
    return weights
