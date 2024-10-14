import numba as nb
import numpy as np

from numba_celltree import utils as ut
from numba_celltree.constants import INITIAL_STACK_LENGTH, MAX_N_VERTEX


def test_grow():
    a = np.arange(10)
    b = ut.grow(a)
    assert b.shape == (20,)
    assert np.array_equal(a, b[:10])

    a = np.arange(20).reshape((10, 2))
    b = ut.grow(a)
    assert b.shape == (20, 2)
    assert np.array_equal(a, b[:10])


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


def test_push_size_increases():
    size = 0
    stack = np.empty(1)
    size_before = stack.size
    stack, size = ut.push(stack, 0, size)
    stack, size = ut.push(stack, 1, size)
    assert stack.size == 2
    assert size_before < stack.size
    stack, size = ut.push(stack, 10, size)
    assert stack.size == 4
    assert np.array_equal(stack[:3], [0, 1, 10])


def test_copy():
    dst = np.zeros(5)
    src = np.arange(5)
    ut.copy(src, dst, 3)
    assert np.array_equal(dst, [0, 1, 2, 0, 0])


def test_double_stack():
    stack = ut.allocate_double_stack()
    assert stack.shape == (INITIAL_STACK_LENGTH, 2)

    size = 1
    stack, size = ut.push_both(stack, 1, 2, size)
    assert size == 2
    assert len(stack) == INITIAL_STACK_LENGTH
    assert np.array_equal(stack[1], (1, 2))

    size = INITIAL_STACK_LENGTH
    stack, size = ut.push_both(stack, 1, 2, size)
    assert len(stack) == INITIAL_STACK_LENGTH * 2
    assert size == INITIAL_STACK_LENGTH + 1
    assert np.array_equal(stack[INITIAL_STACK_LENGTH], (1, 2))

    a, b, size = ut.pop_both(stack, size)
    assert size == INITIAL_STACK_LENGTH
    assert len(stack) == INITIAL_STACK_LENGTH * 2
    assert a == 1
    assert b == 2


def test_triple_stack():
    stack = ut.allocate_triple_stack()
    assert stack.shape == (INITIAL_STACK_LENGTH, 3)

    size = 1
    stack, size = ut.push_triple(stack, 1, 2, 3, size)
    assert size == 2
    assert len(stack) == INITIAL_STACK_LENGTH
    assert np.array_equal(stack[1], (1, 2, 3))

    size = INITIAL_STACK_LENGTH
    stack, size = ut.push_triple(stack, 1, 2, 3, size)
    assert len(stack) == INITIAL_STACK_LENGTH * 2
    assert size == INITIAL_STACK_LENGTH + 1
    assert np.array_equal(stack[INITIAL_STACK_LENGTH], (1, 2, 3))

    a, b, c, size = ut.pop_triple(stack, size)
    assert size == INITIAL_STACK_LENGTH
    assert len(stack) == INITIAL_STACK_LENGTH * 2
    assert a == 1
    assert b == 2
    assert c == 3


# These array is not returned properly to dynamic python. This is OK: these
# arrays are exclusively for internal use to temporarily store values.
@nb.njit
def do_allocate_stack():
    stack = ut.allocate_stack()
    return (stack.size == INITIAL_STACK_LENGTH) and (stack[:5].size == 5)


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
