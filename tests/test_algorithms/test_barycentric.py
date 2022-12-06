import numpy as np
import pytest

from numba_celltree.algorithms import barycentric_triangle as bt
from numba_celltree.algorithms import barycentric_wachspress as bwp
from numba_celltree.constants import Point, Triangle, Vector


def test_interp_edge_case():
    def compute(a, U, p):
        weights = np.zeros(4)
        bwp.interp_edge_case(a, U, p, weights, 0, 1)
        return weights

    a = Point(0.0, 0.0)
    U = Vector(1.0, 0.0)
    assert np.allclose(compute(a, U, Point(0.0, 0.0)), [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(compute(a, U, Point(1.0, 0.0)), [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(compute(a, U, Point(0.25, 0.0)), [0.75, 0.25, 0.0, 0.0])
    assert np.allclose(compute(a, U, Point(0.75, 0.0)), [0.25, 0.75, 0.0, 0.0])


def test_compute_weights_triangle():
    def compute(triangle, point):
        weights = np.zeros(3)
        bt.compute_weights(triangle, point, weights)
        return weights

    triangle = Triangle(
        Point(0.0, 0.0),
        Point(1.0, 0.0),
        Point(1.0, 1.0),
    )

    # Test for the vertices
    assert np.allclose(compute(triangle, Point(0.0, 0.0)), [1.0, 0.0, 0.0])
    assert np.allclose(compute(triangle, Point(1.0, 0.0)), [0.0, 1.0, 0.0])
    assert np.allclose(compute(triangle, Point(1.0, 1.0)), [0.0, 0.0, 1.0])

    # Test halfway edges
    assert np.allclose(compute(triangle, Point(0.5, 0.0)), [0.5, 0.5, 0.0])
    assert np.allclose(compute(triangle, Point(1.0, 0.5)), [0.0, 0.5, 0.5])
    assert np.allclose(compute(triangle, Point(0.5, 0.5)), [0.5, 0.0, 0.5])


@pytest.mark.parametrize(
    "barycentric_weights",
    [bt.barycentric_triangle_weights, bwp.barycentric_wachspress_weights],
)
def test_barycentric_triangle_weights(barycentric_weights):
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 0.5],
            [2.0, 2.0],
        ]
    )
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
        ]
    )
    face_indices = np.array([0, 0, 0, 0, 0, 0, -1])

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )
    actual = barycentric_weights(points, face_indices, faces, vertices)
    assert np.allclose(actual, expected)


def test_compute_weights_wachspress():
    def compute(polygon, point):
        weights = np.zeros(4)
        bwp.compute_weights(polygon, point, weights)
        return weights

    polygon = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    assert np.allclose(compute(polygon, Point(0.0, 0.0)), [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(compute(polygon, Point(1.0, 0.0)), [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(compute(polygon, Point(1.0, 1.0)), [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(compute(polygon, Point(0.0, 1.0)), [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(compute(polygon, Point(0.25, 0.0)), [0.75, 0.25, 0.0, 0.0])
    assert np.allclose(compute(polygon, Point(0.25, 1.0)), [0.0, 0.0, 0.25, 0.75])
    assert np.allclose(compute(polygon, Point(0.5, 0.5)), [0.25, 0.25, 0.25, 0.25])


def test_barycentric_wachspress_weights():
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2, 3],
        ]
    )
    face_indices = np.array([0, 0, 0, 0, 0, 0, 0, -1])
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.25, 0.0],
            [0.25, 1.0],
            [0.5, 0.5],
            [2.0, 2.0],
        ]
    )
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.75, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.75],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    actual = bwp.barycentric_wachspress_weights(points, face_indices, faces, vertices)
    assert np.allclose(actual, expected)
