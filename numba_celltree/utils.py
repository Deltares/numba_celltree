import numpy as np
import numba as nb
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic


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
