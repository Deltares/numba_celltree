import os

import numba as nb
import numpy as np

from numba_celltree import geometry_utils as gu
from numba_celltree.constants import Box, Point, Vector


def test_to_vector():
    a = Point(0.0, 0.0)
    b = Point(1.0, 2.0)
    actual = gu.to_vector(a, b)
    assert isinstance(actual, Vector)
    assert actual.x == 1.0
    assert actual.y == 2.0


def test_as_point():
    a = np.array([0.0, 1.0])
    actual = gu.as_point(a)
    assert isinstance(actual, Point)
    assert actual.x == 0.0
    assert actual.y == 1.0


def test_to_point():
    a = Point(0.0, 0.0)
    b = Point(1.0, 2.0)
    V = gu.to_vector(a, b)
    t = 0.0
    actual = gu.to_point(t, a, V)
    assert np.allclose(actual, a)

    t = 1.0
    actual = gu.to_point(t, a, V)
    assert np.allclose(actual, b)

    t = 0.5
    actual = gu.to_point(t, a, V)
    assert np.allclose(actual, Point(0.5, 1.0))


def test_cross_product():
    u = Vector(1.0, 2.0)
    v = Vector(3.0, 4.0)
    assert np.allclose(gu.cross_product(u, v), np.cross(u, v))


def test_dot_product():
    u = Vector(1.0, 2.0)
    v = Vector(3.0, 4.0)
    assert np.allclose(gu.dot_product(u, v), np.dot(u, v))


def test_point_norm():
    p = Point(0.0, 0.0)
    v = Vector(0.0, 1.0)
    u = Vector(1.0, 0.0)
    actual = gu.point_norm(p, u, v)
    assert np.allclose(actual, Vector(1.0, 1.0))
    actual = gu.point_norm(p, v, u)
    assert np.allclose(actual, Vector(1.0, 1.0))


def test_polygon_length():
    face = np.array([0, 1, 2])
    assert gu.polygon_length(face) == 3
    assert gu.polygon_length(face) == 3
    face = np.array([0, 1, 2, -1, -1])
    assert gu.polygon_length(face) == 3
    face = np.array([0, 1, 2, 3, -1])
    assert gu.polygon_length(face) == 4


def test_polygon_area():
    # square
    p = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    assert np.allclose(gu.polygon_area(p), 1.0)
    # triangle
    p = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    assert np.allclose(gu.polygon_area(p), 0.5)
    # pentagon, counter-clockwise
    p = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.0],
            [0.0, 1.0],
        ]
    )
    assert np.allclose(gu.polygon_area(p), 1.5)
    # clockwise
    assert np.allclose(gu.polygon_area(p[::-1]), 1.5)


def test_point_in_polygon():
    poly = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    assert gu.point_in_polygon(Point(0.5, 0.25), poly)
    assert not gu.point_in_polygon(Point(1.5, 0.25), poly)


def test_boxes_intersect():
    # Identity
    a = Box(0.0, 1.0, 0.0, 1.0)
    b = a
    assert gu.boxes_intersect(a, b)
    assert gu.boxes_intersect(b, a)
    # Overlap
    b = Box(0.5, 1.5, 0.0, 1.0)
    assert gu.boxes_intersect(a, b)
    assert gu.boxes_intersect(b, a)
    # No overlap
    b = Box(1.5, 2.5, 0.5, 1.0)
    assert not gu.boxes_intersect(a, b)
    assert not gu.boxes_intersect(b, a)
    # Different identity
    b = a
    assert gu.boxes_intersect(a, b)
    assert gu.boxes_intersect(b, a)
    # Inside
    a = Box(0.0, 1.0, 0.0, 1.0)
    b = Box(0.25, 0.75, 0.25, 0.75)
    assert gu.boxes_intersect(a, b)
    assert gu.boxes_intersect(b, a)


def test_bounding_box():
    face = np.array([0, 1, 2])
    vertices = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    assert gu.bounding_box(face, vertices) == (0.0, 1.0, 0.0, 1.0)
    face = np.array([0, 1, 2, -1, -1])
    assert gu.bounding_box(face, vertices) == (0.0, 1.0, 0.0, 1.0)


def test_build_bboxes():
    faces = np.array(
        [
            [0, 1, 2, -1],
            [0, 1, 2, 3],
        ]
    )
    vertices = np.array(
        [
            [0.0, 5.0],
            [5.0, 0.0],
            [5.0, 5.0],
            [0.0, 5.0],
        ]
    )
    expected = np.array(
        [
            [0.0, 5.0, 0.0, 5.0],
            [0.0, 5.0, 0.0, 5.0],
        ]
    )
    actual = gu.build_bboxes(faces, vertices)
    assert np.array_equal(actual, expected)


def test_copy_vertices():
    """
    This has to be tested inside of numba jitted function, because the vertices
    are copied to a stack allocated array. This array is not returned properly
    to dynamic python. This is OK: these arrays are exclusively for internal
    use to temporarily store values.
    """
    if os.environ.get("NUMBA_DISABLE_JIT", "0") == "0":

        @nb.njit()
        def test():
            face = np.array([0, 1, 2, -1, -1])
            vertices = np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            )
            expected = vertices.copy()
            actual = gu.copy_vertices(vertices, face)
            result = True
            for i in range(3):
                result = result and actual[i, 0] == expected[i, 0]
                result = result and actual[i, 1] == expected[i, 1]
            return result

        assert test()

    else:
        face = np.array([0, 1, 2, -1, -1])
        vertices = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        expected = vertices.copy()
        actual = gu.copy_vertices(vertices, face)
        assert np.array_equal(actual, expected)
        assert len(actual) == 3


def test_point_inside_box():
    box = Box(0.0, 1.0, 0.0, 1.0)
    a = Point(0.5, 0.5)
    assert gu.point_inside_box(a, box)
    a = Point(-0.5, 0.5)
    assert not gu.point_inside_box(a, box)
    a = Point(0.5, -0.5)
    assert not gu.point_inside_box(a, box)
