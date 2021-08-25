import numpy as np
from numba_celltree.algorithms.separating_axis import (
    polygons_intersect,
    separating_axes,
)


def test_triangles_intersect():
    # Identity
    a = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    b = a
    assert separating_axes(a, b)
    assert separating_axes(b, a)

    # No overlap
    b = np.array(
        [
            [2.0, 0.0],
            [3.0, 1.0],
            [2.0, 1.0],
        ]
    )
    assert not separating_axes(a, b)
    assert not separating_axes(b, a)

    # Touching: does not qualify
    b = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
        ]
    )
    assert not separating_axes(a, b)
    assert not separating_axes(b, a)

    # One inside the other
    a = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
        ]
    )
    b = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    assert separating_axes(a, b)
    assert separating_axes(b, a)

    # Mirrored
    a = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    b = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    assert not separating_axes(a, b)
    assert not separating_axes(b, a)


def test_box_triangle():
    a = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )
    b = np.array(
        [
            [1.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ]
    )
    assert separating_axes(a, b)
    assert separating_axes(b, a)

    # Touching
    b = np.array(
        [
            [2.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ]
    )
    assert not separating_axes(a, b)
    assert not separating_axes(b, a)

    # Inside
    b = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.75],
            [0.6, 0.6],
        ]
    )
    assert separating_axes(a, b)
    assert separating_axes(b, a)


def test_polygons_intersect():
    vertices_a = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )
    faces_a = np.array(
        [
            [0, 1, 2, 3],
        ]
    )
    vertices_b = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.75],
            [0.6, 0.6],
            [2.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ]
    )
    faces_b = np.array(
        [
            [0, 1, 2, -1],
            [3, 4, 5, -1],
        ]
    )
    indices_a = np.array([0, 0])
    indices_b = np.array([0, 1])

    actual = polygons_intersect(
        vertices_a, vertices_b, faces_a, faces_b, indices_a, indices_b
    )
    expected = np.array([True, False])
    assert np.array_equal(actual, expected)


def test_triangles_intersect_hanging_nodes():
    # Identity
    a = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    b = a
    assert separating_axes(a, b)
    assert separating_axes(b, a)

    # No overlap
    b = np.array(
        [
            [2.0, 0.0],
            [3.0, 1.0],
            [2.5, 1.0],
            [2.0, 1.0],
        ]
    )
    assert not separating_axes(a, b)
    assert not separating_axes(b, a)
