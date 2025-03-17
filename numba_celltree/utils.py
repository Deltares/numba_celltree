import os

import numba as nb
import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

from numba_celltree.constants import (
    INITIAL_STACK_LENGTH,
    MAX_N_VERTEX,
    NDIM,
    STACK_ALLOCATE_STATIC_ARRAYS,
    FloatDType,
    IntDType,
)


@nb.njit(inline="always")
def allocate_stack():
    return np.empty(INITIAL_STACK_LENGTH, dtype=IntDType)


@nb.njit(inline="always")
def allocate_double_stack():
    return np.empty((INITIAL_STACK_LENGTH, 2), dtype=IntDType)


@nb.njit(inline="always")
def allocate_triple_stack():
    return np.empty((INITIAL_STACK_LENGTH, 3), dtype=IntDType)


@nb.njit(inline="always")
def pop(array, size):
    return array[size - 1], size - 1


@nb.njit(inline="always")
def push(array, value, size):
    if size >= len(array):
        array = grow(array)
    array[size] = value
    return array, size + 1


@nb.njit(inline="always")
def copy(src, dst, n) -> None:
    for i in range(n):
        dst[i] = src[i]
    return


@nb.njit(inline="always")
def push_both(stack, a, b, size):
    if size >= len(stack):
        stack = grow(stack)
    stack[size, 0] = a
    stack[size, 1] = b
    return stack, size + 1


@nb.njit(inline="always")
def pop_both(stack, size):
    index = size - 1
    a = stack[index, 0]
    b = stack[index, 1]
    return a, b, index


@nb.njit(inline="always")
def push_triple(stack, a, b, c, size):
    if size >= len(stack):
        stack = grow(stack)
    stack[size, 0] = a
    stack[size, 1] = b
    stack[size, 2] = c
    return stack, size + 1


@nb.njit(inline="always")
def pop_triple(stack, size):
    index = size - 1
    a = stack[index, 0]
    b = stack[index, 1]
    c = stack[index, 2]
    return a, b, c, index


@nb.njit(inline="always")
def grow(array: np.ndarray) -> np.ndarray:
    """Double storage capacity."""
    n = len(array)
    new_shape = (2 * n,) + array.shape[1:]
    new = np.empty(new_shape, dtype=array.dtype)
    new[:n] = array[:]
    return new


# Ensure these are constants for numba
POLYGON_SIZE = MAX_N_VERTEX * NDIM
CLIP_MAX_N_VERTEX = MAX_N_VERTEX * 2
CLIP_POLYGON_SIZE = 2 * POLYGON_SIZE


# Make sure everything still works when calling as non-compiled Python code.
# Note: these stack allocated arrays should only be used inside of numba
# compiled code. They should interact NEVER with dynamic Python code: there are
# no guarantees in that case, they may very well be filled with garbage.
if STACK_ALLOCATE_STATIC_ARRAYS and os.environ.get("NUMBA_DISABLE_JIT", "0") == "0":

    @intrinsic  # pragma: no cover
    def stack_empty(typingctx, size, dtype):
        def impl(context, builder, signature, args):
            ty = context.get_value_type(dtype.dtype)
            ptr = cgutils.alloca_once(builder, ty, size=args[0])
            return ptr

        sig = types.CPointer(dtype.dtype)(types.int64, dtype)
        return sig, impl

    @nb.njit(inline="always")  # pragma: no cover
    def allocate_polygon():
        arr_ptr = stack_empty(  # pylint: disable=no-value-for-parameter
            POLYGON_SIZE, FloatDType
        )
        arr = nb.carray(arr_ptr, (MAX_N_VERTEX, NDIM), dtype=FloatDType)
        return arr

    @nb.njit(inline="always")  # pragma: no cover
    def allocate_clip_polygon():
        arr_ptr = stack_empty(  # pylint: disable=no-value-for-parameter
            CLIP_POLYGON_SIZE, FloatDType
        )
        arr = nb.carray(arr_ptr, (CLIP_MAX_N_VERTEX, NDIM), dtype=FloatDType)
        return arr

    @nb.njit(inline="always")  # pragma: no cover
    def allocate_box_polygon():
        arr_ptr = stack_empty(8, FloatDType)  # pylint: disable=no-value-for-parameter
        arr = nb.carray(arr_ptr, (4, 2), dtype=FloatDType)
        return arr

else:

    @nb.njit(inline="always")
    def allocate_polygon():
        return np.empty((MAX_N_VERTEX, NDIM), dtype=FloatDType)

    @nb.njit(inline="always")
    def allocate_clip_polygon():
        return np.empty((CLIP_MAX_N_VERTEX, NDIM), dtype=FloatDType)

    @nb.njit(inline="always")
    def allocate_box_polygon():
        return np.empty((4, 2), dtype=FloatDType)
