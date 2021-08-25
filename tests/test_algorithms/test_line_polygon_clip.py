"""
Utilizes same boxes as test_line_box_clip
"""
import numpy as np

from numba_celltree.algorithms import cyrus_beck_line_polygon_clip
from numba_celltree.constants import Point


def test_line_box_clip():
    line_clip = cyrus_beck_line_polygon_clip

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

    a = Point(0.0, 0.1)
    b = Point(0.0, 0.1)
    intersects, c, d = line_clip(a, b, poly)
    assert not intersects
    assert np.isnan(c).all()
    assert np.isnan(d).all()

    a = Point(0.0, 4.0)
    b = Point(5.0, 4.0)
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.0, 4.0])
    assert np.allclose(d, [4.0, 4.0])

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

    b = Point(1.0, 1.0)
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.0, 0.0])
    assert np.allclose(d, [1.0, 1.0])

    a = Point(1.0, 1.0)
    b = Point(1.0, 3.0)
    intersects, c, d = line_clip(a, b, poly)
    assert intersects
    assert np.allclose(c, [1.0, 1.0])
    assert np.allclose(d, [1.0, 2.0])
