"""
Compute the Wachspress Barycentric ccoordinate for a convex polygon. These can
be directly used for barycentric interpolation.
"""
import numba as nb
import numpy as np

from ..constants import PARALLEL, TOLERANCE_ON_EDGE, FloatArray, FloatDType, IntArray
from ..geometry_utils import (
    Point,
    as_point,
    copy_vertices,
    cross_product,
    dot_product,
    to_vector,
)


@nb.njit(inline="always")
def interp_edge_case(a, U, p, weights, i, j):
    # For the edge case, find the linear interpolation weight between the
    # vertices.
    weights[:] = 0
    V = to_vector(a, p)
    w = np.sqrt(dot_product(V, V)) / np.sqrt(dot_product(U, U))
    weights[i] = 1.0 - w
    weights[j] = w
    return


@nb.njit
def compute_weights(polygon: FloatArray, p: Point, weights: FloatArray):
    n = len(polygon)
    w_sum = 0.0

    # Initial iteration
    a = as_point(polygon[-1])
    b = as_point(polygon[0])
    U = to_vector(a, b)
    V = to_vector(a, p)
    Ai = abs(cross_product(U, V))
    if Ai < TOLERANCE_ON_EDGE:
        # Note: weights may be differently sized than polygon! Hence n-1
        # instead of -1.
        interp_edge_case(a, U, p, weights, n - 1, 0)
        return weights

    for i in range(n):
        i_next = (i + 1) % n
        c = as_point(polygon[i_next])

        V = to_vector(a, c)
        Ci = abs(cross_product(U, V))

        U = to_vector(b, p)
        V = to_vector(b, c)
        Aj = abs(cross_product(U, V))

        if Aj < TOLERANCE_ON_EDGE:
            interp_edge_case(b, V, p, weights, i, i_next)
            return weights

        w = 2 * Ci / (Ai * Aj)
        weights[i] = w
        w_sum += w

        # Setup next iteration
        a = b
        b = c
        U = to_vector(a, b)
        Ai = Aj

    # normalize weights
    for i in range(n):
        weights[i] /= w_sum

    return


@nb.njit(parallel=PARALLEL, cache=True)
def barycentric_wachspress_weights(
    points: FloatArray,
    face_indices: IntArray,
    faces: IntArray,
    vertices: FloatArray,
) -> FloatArray:
    n_points = len(points)
    n_max_vert = faces.shape[1]
    weights = np.zeros((n_points, n_max_vert), dtype=FloatDType)
    for i in nb.prange(n_points):
        face_index = face_indices[i]
        if face_index == -1:
            continue
        face = faces[face_index]
        polygon = copy_vertices(vertices, face)
        point = as_point(points[i])
        compute_weights(polygon, point, weights[i])
    return weights
