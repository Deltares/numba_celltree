"""
Compute the Wachspress Barycentric ccoordinate for a convex polygon. These can
be directly used for barycentric interpolation.
"""
import numba as nb
import numpy as np

from ..constants import TOLERANCE_ON_EDGE, FloatArray, FloatDType
from ..geometry_utils import Point, as_point, cross_product, dot_product, to_vector


@nb.njit(inline="always")
def edge_case(a, U, p, weights, i, j):
    # For the edge case, find the linear interpolation weight between the
    # vertices.
    weights[:] = 0
    V = to_vector(a, p)
    w = np.sqrt(dot_product(V, V)) / np.sqrt(dot_product(U, U))
    weights[i] = w
    weights[j] = 1.0 - w
    return


@nb.njit
def wachspress_weights(polygon: FloatArray, p: Point):
    n = len(polygon)
    weights = np.empty(n, dtype=FloatDType)
    w_sum = 0.0

    # Initial iteration
    a = as_point(polygon[-1])
    b = as_point(polygon[0])
    U = to_vector(a, b)
    V = to_vector(a, p)
    Ai = abs(cross_product(U, V))
    if Ai < TOLERANCE_ON_EDGE:
        edge_case(a, U, p, weights, -1, 0)
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
            edge_case(b, V, p, weights, i, i_next)
            return weights

        w = 2 * Ci / (Ai * Aj)
        weights[i] = w
        w_sum += w

        # Setup next iteration
        a = b
        b = c
        U = to_vector(a, b)
        Ai = Aj

    return weights / w_sum
