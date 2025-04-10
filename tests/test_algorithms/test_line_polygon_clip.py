"""Utilizes same boxes as test_line_box_clip"""

import numpy as np

from numba_celltree.algorithms.cyrus_beck import cyrus_beck_line_polygon_clip
from numba_celltree.constants import Point


def ab(a, b, c):
    """Flip the result around to compare (a, b) with (b, a)"""
    return (a, c, b)


TOLERANCE = 1e-9


def line_clip(a, b, poly):
    return cyrus_beck_line_polygon_clip(a, b, poly, TOLERANCE)


def test_line_box_clip():
    poly = np.array(
        [
            [1.0, 3.0],
            [4.0, 3.0],
            [4.0, 5.0],
            [1.0, 5.0],
        ]
    )

    a = Point(0.0, 0.0)
    b = Point(4.0, 6.0)
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [2.0, 3.0])
    assert np.allclose(d, [3.3333333333333, 5.0])
    assert line_clip(a, b, poly) == ab(*line_clip(b, a, poly))

    a = Point(0.0, 0.1)
    b = Point(0.0, 0.1)
    intersects, c, d = line_clip(a, b, poly)
    assert not intersects
    assert np.isnan(c).all()
    assert np.isnan(d).all()
    assert line_clip(a, b, poly)[0] == line_clip(b, a, poly)[0]

    a = Point(0.0, 4.0)
    b = Point(5.0, 4.0)
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.0, 4.0])
    assert np.allclose(d, [4.0, 4.0])
    assert line_clip(a, b, poly) == ab(*line_clip(b, a, poly))

    poly = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )
    a = Point(1.0, -3.0)
    b = Point(1.0, 3.0)
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.0, 0.0])
    assert np.allclose(d, [1.0, 2.0])
    assert line_clip(a, b, poly) == ab(*line_clip(b, a, poly))

    b = Point(1.0, 1.0)
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.0, 0.0])
    assert np.allclose(d, [1.0, 1.0])
    assert line_clip(a, b, poly) == ab(*line_clip(b, a, poly))

    a = Point(1.0, 1.0)
    b = Point(1.0, 3.0)
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.0, 1.0])
    assert np.allclose(d, [1.0, 2.0])
    assert line_clip(a, b, poly) == ab(*line_clip(b, a, poly))


def test_line_box_clip_through_vertex():
    a = Point(x=2.0, y=2.0)
    b = Point(x=1.0, y=4.0)
    poly = np.array([[2.0, 4.0], [1.0, 4.0], [1.0, 3.0], [2.0, 3.0]])
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.5, 3.0])
    assert np.allclose(d, [1.0, 4.0])

    a = Point(x=2.0, y=0.0)
    b = Point(x=2.0, y=2.0)
    poly = np.array([[2.0, 3.0], [1.0, 3.0], [1.0, 2.0], [2.0, 2.0]])
    intersects, _, _ = line_clip(a, b, poly)
    assert not intersects

    poly = np.array([[3.0, 3.0], [2.0, 3.0], [2.0, 2.0], [3.0, 2.0]])
    intersects, c, d = line_clip(a, b, poly)
    assert not intersects


def test_line_triangle_clip():
    # Triangle
    a = Point(1.0, 1.0)
    b = Point(3.0, 1.0)
    poly = np.array(
        [
            [0.0, 0.5],
            [2.0, 0.0],
            [2.0, 2.0],
        ]
    )
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.0, 1.0])
    assert np.allclose(d, [2.0, 1.0])
    assert line_clip(a, b, poly) == ab(*line_clip(b, a, poly))


def test_line_triangle_clip_degeneracies():
    poly = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
        ]
    )
    # Lower edge
    a = Point(0.0, 0.0)
    b = Point(2.0, 0.0)
    assert line_clip(a, b, poly)[0]
    assert line_clip(b, a, poly)[0]

    # Right edge
    a = Point(2.0, 0.0)
    b = Point(2.0, 2.0)
    assert line_clip(a, b, poly)[0]
    assert line_clip(b, a, poly)[0]

    # Diagonal edge
    a = Point(0.0, 0.0)
    b = Point(2.0, 2.0)
    assert line_clip(a, b, poly)[0]
    assert line_clip(b, a, poly)[0]

    a = Point(-1.0, -1.0)
    b = Point(2.0, 2.0)
    assert line_clip(a, b, poly)[0]
    assert line_clip(b, a, poly)[0]

    a = Point(-1.0, -1.0)
    b = Point(3.0, 3.0)
    assert line_clip(a, b, poly)[0]
    assert line_clip(b, a, poly)[0]

    a = Point(0.0, 0.0)
    b = Point(3.0, 3.0)
    assert line_clip(a, b, poly)[0]
    assert line_clip(b, a, poly)[0]


def test_line_box_degeneracies():
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )
    faces = np.array(
        [
            [3, 4, 6, 7],
            [4, 5, 8, 7],
            [0, 1, 4, 3],
            [1, 2, 5, 4],
        ]
    )
    poly = nodes[faces[1]]
    a = Point(1.0, 1.0)
    b = Point(1.5, 1.25)
    succes, c, d = line_clip(a, b, poly)
    assert succes
    assert c == Point(1.0, 1.0)
    assert d == Point(1.5, 1.25)

    for i in [0, 2, 3]:
        poly = nodes[faces[i]]
        a = Point(1.0, 1.0)
        b = Point(1.5, 1.25)
        succes, c, d = line_clip(a, b, poly)
        assert not succes

    # This is a case where a is inside, but b is right on the vertex
    # point_in_poly(a, poly) == True
    # point_in_poly(b, poly) == False
    # This results in t0 == t1 in Cyrus-Beck. However, we can get the right
    # by setting t0 to 0.0 if a_inside, or t1 to 1.0 if b_inside.
    poly = nodes[faces[2]]
    a = Point(0.5, 0.75)
    b = Point(1.0, 1.0)
    succes, c, d = line_clip(a, b, poly)
    assert succes
    assert c == a
    assert d == b
