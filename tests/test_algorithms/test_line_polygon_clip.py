"""
Utilizes same boxes as test_line_box_clip
"""
import numpy as np

from numba_celltree.algorithms import cyrus_beck_line_polygon_clip as line_clip
from numba_celltree.constants import Point


def ab(a, b, c):
    """Flip the result around to compare (a, b) with (b, a)"""
    return (a, c, b)


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
