import numba as nb
import numpy as np

from numba_celltree import utils as ut
from numba_celltree.constants import INITIAL_TREE_DEPTH, MAX_N_VERTEX


def test_pop():
    size = 3
    stack = np.arange(10, 10 + size)
    v, size = ut.pop(stack, size)
    assert v == 12
    assert size == 2
    v, size = ut.pop(stack, size)
    assert v == 11
    assert size == 1
    v, size = ut.pop(stack, size)
    assert v == 10
    assert size == 0


def test_push():
    size = 0
    stack = np.empty(3)
    stack, size = ut.push(stack, 0, size)
    stack, size = ut.push(stack, 1, size)
    stack, size = ut.push(stack, 2, size)
    assert size == 3
    assert np.array_equal(stack, [0, 1, 2])


def test_copy():
    dst = np.zeros(5)
    src = np.arange(5)
    ut.copy(src, dst, 3)
    assert np.array_equal(dst, [0, 1, 2, 0, 0])


# These array is not returned properly to dynamic python. This is OK: these
# arrays are exclusively for internal use to temporarily store values.
@nb.njit
def do_allocate_stack():
    stack = ut.allocate_stack()
    return (stack.size == INITIAL_TREE_DEPTH) and (stack[:5].size == 5)


def test_allocate_stack():
    assert do_allocate_stack()


@nb.njit
def do_allocate_polygon():
    poly = ut.allocate_polygon()
    return poly.shape == (MAX_N_VERTEX, 2) and (poly[:5].size == 10)


def test_allocate_polygon():
    assert do_allocate_polygon()


@nb.njit
def do_allocate_clipper():
    clipper = ut.allocate_clip_polygon()
    return clipper.shape == (MAX_N_VERTEX * 2, 2) and (clipper[:5].size == 10)


def test_allocate_clipper():
    assert do_allocate_clipper()
