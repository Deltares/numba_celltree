import os

import numba as nb
import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

from .constants import MAX_TREE_DEPTH, IntDType


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
def reallocate(a):
    b = np.empty(2 * a.size, a.dtype)
    b[: a.size] = a
    return b


@nb.njit(inline="always")
def push_amortized(array, value, size):
    if size >= array.size:
        array = reallocate(array)
    array[size] = value
    return array, size + 1


def np_allocate_stack():
    return np.empty(MAX_TREE_DEPTH, dtype=IntDType)


@nb.njit(inline="always")
def nb_allocate_stack():
    arr_ptr = stack_empty(
        MAX_TREE_DEPTH, IntDType
    )  # pylint: disable=no-value-for-parameter
    arr = nb.carray(arr_ptr, MAX_TREE_DEPTH, dtype=IntDType)
    return arr


# Make sure everything still works when calling as non-compiled Python code:
if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1":
    allocate_stack = np_allocate_stack
else:
    allocate_stack = nb_allocate_stack
