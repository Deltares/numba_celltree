import os

import numba as nb
import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

from .constants import MAX_N_VERTEX, MAX_TREE_DEPTH, NDIM, FloatDType, IntDType


@intrinsic
def stack_empty(typingctx, size, dtype):
    def impl(context, builder, signature, args):
        ty = context.get_value_type(dtype.dtype)
        ptr = cgutils.alloca_once(builder, ty, size=args[0])
        return ptr

    sig = types.CPointer(dtype.dtype)(types.int64, dtype)
    return sig, impl


@nb.njit(inline="always")
def pop(array, size):
    return array[size - 1], size - 1


@nb.njit(inline="always")
def push(array, value, size):
    array[size] = value
    return size + 1


@nb.njit(inline="always")
def copy(src, dst, n) -> None:
    for i in range(n):
        dst[i] = src[i]


def np_allocate_stack():
    return np.empty(MAX_TREE_DEPTH, dtype=IntDType)


# Ensure these are constants for numba
POLYGON_SIZE = MAX_N_VERTEX * NDIM
CLIP_MAX_N_VERTEX = MAX_N_VERTEX * 2
CLIP_POLYGON_SIZE = 2 * POLYGON_SIZE

# Note: these stack allocated arrays should only be used inside of numba
# compiled code. They should interact NEVER with dynamic Python code: there are
# no guarantees in that case, they may very well be filled with garbage.


@nb.njit(inline="always")
def nb_allocate_stack():
    arr_ptr = stack_empty(
        MAX_TREE_DEPTH, IntDType
    )  # pylint: disable=no-value-for-parameter
    arr = nb.carray(arr_ptr, MAX_TREE_DEPTH, dtype=IntDType)
    return arr


def np_allocate_polygon():
    return np.empty((MAX_N_VERTEX, NDIM), dtype=FloatDType)


@nb.njit(inline="always")
def nb_allocate_polygon():
    arr_ptr = stack_empty(  # pylint: disable=no-value-for-parameter
        POLYGON_SIZE, FloatDType
    )
    arr = nb.carray(arr_ptr, (MAX_N_VERTEX, NDIM), dtype=FloatDType)
    return arr


def np_allocate_clip_polygon():
    return np.empty((CLIP_MAX_N_VERTEX, NDIM), dtype=FloatDType)


@nb.njit(inline="always")
def nb_allocate_clip_polygon():
    arr_ptr = stack_empty(  # pylint: disable=no-value-for-parameter
        CLIP_POLYGON_SIZE, FloatDType
    )
    arr = nb.carray(arr_ptr, (CLIP_MAX_N_VERTEX, NDIM), dtype=FloatDType)
    return arr


# Make sure everything still works when calling as non-compiled Python code:
if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1":
    allocate_stack = np_allocate_stack
    allocate_polygon = np_allocate_polygon
    allocate_clip_polygon = np_allocate_clip_polygon
else:
    allocate_stack = nb_allocate_stack
    allocate_polygon = nb_allocate_polygon
    allocate_clip_polygon = nb_allocate_clip_polygon
