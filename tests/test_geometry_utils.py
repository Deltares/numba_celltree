import os

import numba as nb
import numpy as np

from numba_celltree import geometry_utils as gu
from numba_celltree.constants import TOLERANCE_ON_EDGE, Box, Point, Triangle, Vector


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


def test_as_box():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    actual = gu.as_box(a)
    assert isinstance(actual, Box)
    assert actual.xmin == 0.0
    assert actual.xmax == 1.0
    assert actual.ymin == 2.0
    assert actual.ymax == 3.0


def test_as_triangle():
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    face = np.array([2, 0, 1])
    actual = gu.as_triangle(vertices, face)
    assert isinstance(actual, Triangle)
    assert actual.a == Point(1.0, 1.0)
    assert actual.b == Point(0.0, 0.0)
    assert actual.c == Point(1.0, 0.0)


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
    assert np.allclose(gu.cross_product(u, v), u.x * v.y - u.y * v.x)


def test_dot_product():
    u = Vector(1.0, 2.0)
    v = Vector(3.0, 4.0)
    assert np.allclose(gu.dot_product(u, v), np.dot(u, v))


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

    assert gu.point_in_polygon(Point(0.0, 0.0), poly)
    assert gu.point_in_polygon(Point(0.0, 0.0), poly[::-1])
    assert gu.point_in_polygon(Point(0.5, 0.5), poly)
    assert gu.point_in_polygon(Point(0.5, 0.5), poly[::-1])
    assert not gu.point_in_polygon(Point(1.0, 1.0), poly)
    assert not gu.point_in_polygon(Point(1.0, 1.0), poly[::-1])


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


def test_box_contained():
    a = Box(0.0, 1.0, 0.0, 1.0)
    b = Box(0.25, 0.75, 0.25, 0.75)
    assert gu.box_contained(a, a)
    assert gu.box_contained(b, a)
    assert not gu.box_contained(a, b)


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


def test_build_face_bboxes():
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
    actual = gu.build_face_bboxes(faces, vertices)
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


def test_copy_vertices_into():
    out = np.empty((10, 2))
    face = np.array([0, 1, 2, -1, -1])
    vertices = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    expected = vertices.copy()
    actual = gu.copy_vertices_into(vertices, face, out)
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


def test_flip():
    face0 = np.array([0, 1, 2, -1, -1])
    face1 = np.array([0, 1, 2, 3, -1])
    face2 = np.array([0, 1, 2, 3, 4])
    gu.flip(face0, 3)
    gu.flip(face1, 4)
    gu.flip(face2, 5)
    assert np.array_equal(face0, [2, 1, 0, -1, -1])
    assert np.array_equal(face1, [3, 2, 1, 0, -1])
    assert np.array_equal(face2, [4, 3, 2, 1, 0])


def test_counter_clockwise():
    vertices = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],  # hanging node
            [1.0, 0.0],
            [1.0, 0.5],  # hanging node
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    ccw_faces = np.array(
        [
            [0, 2, 4, 5, -1, -1],
            [0, 1, 2, 3, 4, 5],
        ]
    )
    cw_faces = np.array(
        [
            [5, 4, 2, 0, -1, -1],
            [5, 4, 3, 2, 1, 0],
        ]
    )
    expected = ccw_faces.copy()
    # already counter clockwise should not be mutated
    gu.counter_clockwise(vertices, ccw_faces)
    assert np.array_equal(expected, ccw_faces)
    # clockwise should be mutated
    gu.counter_clockwise(vertices, cw_faces)
    assert np.array_equal(expected, cw_faces)


def test_lines_intersect():
    offset = 2 * TOLERANCE_ON_EDGE
    a = Point(0.0, 0.0)
    b = Point(4.0, 4.0)
    # Edges not intersecting
    p = Point(3.0, 2.0)
    q = Point(3.0, 1.0)
    assert not gu.lines_intersect(a, b, p, q)
    assert not gu.lines_intersect(p, q, a, b)
    # Vertex nearly touching edge
    p = Point(3.0, 3.0 - offset)
    q = Point(3.0, 1.0)
    assert not gu.lines_intersect(a, b, p, q)
    assert not gu.lines_intersect(p, q, a, b)
    # Vertex on edge
    p = Point(3.0, 3.0)
    q = Point(3.0, 1.0)
    assert gu.lines_intersect(a, b, p, q)
    assert gu.lines_intersect(p, q, a, b)
    # Edge on edge, collinear (fully overlapping)
    p = Point(1.0, 1.0)
    q = Point(3.0, 3.0)
    assert gu.lines_intersect(a, b, p, q)
    assert gu.lines_intersect(p, q, a, b)
    # Edge on edge, orthogonal
    p = Point(1.0, 3.0)
    q = Point(3.0, 1.0)
    assert gu.lines_intersect(a, b, p, q)
    assert gu.lines_intersect(p, q, a, b)
    # Vertex on edge
    p = Point(-1.0, 1.0)
    q = Point(1.0, -1.0)
    assert gu.lines_intersect(a, b, p, q)
    assert gu.lines_intersect(p, q, a, b)
    # Vertex on vertex
    p = Point(0.0, 0.0)
    q = Point(1.0, -1.0)
    assert gu.lines_intersect(a, b, p, q)
    assert gu.lines_intersect(p, q, a, b)
    # Vertex on vertex, collinear
    p = Point(0.0, 0.0)
    q = Point(-1.0, -1.0)
    assert gu.lines_intersect(a, b, p, q)
    assert gu.lines_intersect(p, q, a, b)
    # Vertex nearly on vertex, collinear, no overlap
    p = Point(-offset, -offset)
    q = Point(-1.0, -1.0)
    assert not gu.lines_intersect(a, b, p, q)
    assert not gu.lines_intersect(p, q, a, b)
