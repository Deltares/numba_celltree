import numpy as np
import pytest

from numba_celltree.algorithms import (
    cohen_sutherland_line_box_clip,
    cyrus_beck_line_polygon_clip,
    liang_barsky_line_box_clip,
)
from numba_celltree.constants import Box, Point


def ab(a, b, c):
    """Flip the result around to compare (a, b) with (b, a)"""
    return (a, c, b)


BOX = Box(0.0, 2.0, 0.0, 2.0)
POLY = np.array(
    [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
    ]
)


@pytest.mark.parametrize(
    "line_clip, box",
    [
        (cohen_sutherland_line_box_clip, BOX),
        (liang_barsky_line_box_clip, BOX),
        (cyrus_beck_line_polygon_clip, POLY),
    ],
)
def test_line_box_clip(line_clip, box):
    a = Point(-1.0, 0.0)
    b = Point(2.0, 3.0)
    intersects, c, d = line_clip(a, b, box)
    assert intersects
    assert np.allclose(c, [0.0, 1.0])
    assert np.allclose(d, [1.0, 2.0])

    a = Point(0.0, -0.1)
    b = Point(0.0, -0.1)
    intersects, c, d = line_clip(a, b, box)
    assert not intersects
    assert np.isnan(c).all()
    assert np.isnan(d).all()

    a = Point(-1.0, 1.0)
    b = Point(3.0, 1.0)
    intersects, c, d = line_clip(a, b, box)
    assert intersects
    assert np.allclose(c, [0.0, 1.0])
    assert np.allclose(d, [2.0, 1.0])

    a = Point(1.0, -3.0)
    b = Point(1.0, 3.0)
    intersects, c, d = line_clip(a, b, box)
    assert intersects
    assert np.allclose(c, [1.0, 0.0])
    assert np.allclose(d, [1.0, 2.0])

    b = Point(1.0, 1.0)
    intersects, c, d = line_clip(a, b, box)
    assert intersects
    assert np.allclose(c, [1.0, 0.0])
    assert np.allclose(d, [1.0, 1.0])

    a = Point(1.0, 1.0)
    b = Point(1.0, 3.0)
    intersects, c, d = line_clip(a, b, box)
    assert intersects
    assert np.allclose(c, [1.0, 1.0])
    assert np.allclose(d, [1.0, 2.0])

    a = Point(-1.0, 3.0)
    b = Point(3.0, 3.0)
    intersects, c, d = line_clip(a, b, box)
    assert not intersects

    a = Point(-1.0, 1.0)
    b = Point(1.0, 1.0)
    intersects, c, d = line_clip(a, b, box)
    assert intersects
    assert np.allclose(c, [0.0, 1.0])
    assert np.allclose(d, [1.0, 1.0])

    # both inside
    a = Point(0.5, 0.5)
    b = Point(1.5, 1.5)
    intersects, c, d = line_clip(a, b, box)
    assert intersects
    assert np.allclose(c, a)
    assert np.allclose(d, b)

    # No intersection, left
    a = Point(-1.5, 0.0)
    b = Point(-0.5, 1.0)
    intersects, c, d = line_clip(a, b, box)
    assert not intersects

    # No intersection, right
    a = Point(2.5, 0.0)
    b = Point(3.5, 1.0)
    intersects, c, d = line_clip(a, b, box)
    assert not intersects


@pytest.mark.parametrize(
    "line_clip, box",
    [
        (cohen_sutherland_line_box_clip, BOX),
        (liang_barsky_line_box_clip, BOX),
        (cyrus_beck_line_polygon_clip, POLY),
    ],
)
def test_line_box_clip_degeneracy(line_clip, box):
    # Line through vertices
    a = Point(-1.0, -1.0)
    b = Point(3.0, 3.0)
    intersects, c, d = line_clip(a, b, box)
    assert intersects
    assert np.allclose(c, [0.0, 0.0])
    assert np.allclose(d, [2.0, 2.0])
    assert line_clip(a, b, box)[0] == line_clip(b, a, box)[0]

    # Identity
    intersects, c, d = line_clip(a, a, box)
    assert not intersects

    # Line through lower edge
    a = Point(-1.0, 0.0)
    b = Point(3.0, 0.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Line through upper edge
    a = Point(-1.0, 2.0)
    b = Point(3.0, 2.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Partial line lower edge
    a = Point(-1.0, 0.0)
    b = Point(1.0, 0.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Partial line upper edge
    a = Point(-1.0, 2.0)
    b = Point(1.0, 2.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Within lower edge
    a = Point(0.5, 0.0)
    b = Point(1.5, 0.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Within upper edge
    a = Point(0.5, 2.0)
    b = Point(1.5, 2.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Identical to lower edge
    a = Point(0.0, 0.0)
    b = Point(2.0, 0.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Identical to upper edge
    a = Point(0.0, 2.0)
    b = Point(2.0, 2.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Identical to left edge
    a = Point(0.0, 0.0)
    b = Point(0.0, 2.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Identical to right edge
    a = Point(2.0, 0.0)
    b = Point(2.0, 2.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Within left edge
    a = Point(0.0, 0.5)
    b = Point(0.0, 1.5)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Within right edge
    a = Point(2.0, 0.5)
    b = Point(2.0, 1.5)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Identical to left edge
    a = Point(0.0, 0.0)
    b = Point(0.0, 2.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]

    # Identical to right edge
    a = Point(2.0, 0.0)
    b = Point(2.0, 2.0)
    assert line_clip(a, b, box)[0]
    assert line_clip(b, a, box)[0]
