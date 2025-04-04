import numpy as np
import pytest

from numba_celltree.algorithms import (
    cohen_sutherland_line_box_clip,
    cyrus_beck_line_polygon_clip,
    liang_barsky_line_box_clip,
)
from numba_celltree.constants import TOLERANCE_ON_EDGE, Box, Point


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
POLY_REVERSED = POLY[::-1, :]


@pytest.mark.parametrize(
    "line_clip, args",
    [
        (cohen_sutherland_line_box_clip, (BOX,)),
        (liang_barsky_line_box_clip, (BOX,)),
        (cyrus_beck_line_polygon_clip, (POLY,)),
    ],
)
def test_line_box_clip(line_clip, args):
    a = Point(-1.0, 0.0)
    b = Point(2.0, 3.0)
    intersects, c, d = line_clip(a, b, *args)
    assert intersects
    assert np.allclose(c, [0.0, 1.0])
    assert np.allclose(d, [1.0, 2.0])

    a = Point(0.0, -0.1)
    b = Point(0.0, -0.1)
    intersects, c, d = line_clip(a, b, *args)
    assert not intersects
    assert np.isnan(c).all()
    assert np.isnan(d).all()

    a = Point(-1.0, 1.0)
    b = Point(3.0, 1.0)
    intersects, c, d = line_clip(a, b, *args)
    assert intersects
    assert np.allclose(c, [0.0, 1.0])
    assert np.allclose(d, [2.0, 1.0])

    a = Point(1.0, -3.0)
    b = Point(1.0, 3.0)
    intersects, c, d = line_clip(a, b, *args)
    assert intersects
    assert np.allclose(c, [1.0, 0.0])
    assert np.allclose(d, [1.0, 2.0])

    b = Point(1.0, 1.0)
    intersects, c, d = line_clip(a, b, *args)
    assert intersects
    assert np.allclose(c, [1.0, 0.0])
    assert np.allclose(d, [1.0, 1.0])

    a = Point(1.0, 1.0)
    b = Point(1.0, 3.0)
    intersects, c, d = line_clip(a, b, *args)
    assert intersects
    assert np.allclose(c, [1.0, 1.0])
    assert np.allclose(d, [1.0, 2.0])

    a = Point(-1.0, 3.0)
    b = Point(3.0, 3.0)
    intersects, c, d = line_clip(a, b, *args)
    assert not intersects

    a = Point(-1.0, 1.0)
    b = Point(1.0, 1.0)
    intersects, c, d = line_clip(a, b, *args)
    assert intersects
    assert np.allclose(c, [0.0, 1.0])
    assert np.allclose(d, [1.0, 1.0])

    # both inside
    a = Point(0.5, 0.5)
    b = Point(1.5, 1.5)
    intersects, c, d = line_clip(a, b, *args)
    assert intersects
    assert np.allclose(c, a)
    assert np.allclose(d, b)

    # No intersection, left
    a = Point(-1.5, 0.0)
    b = Point(-0.5, 1.0)
    intersects, c, d = line_clip(a, b, *args)
    assert not intersects

    # No intersection, right
    a = Point(2.5, 0.0)
    b = Point(3.5, 1.0)
    intersects, c, d = line_clip(a, b, *args)
    assert not intersects


@pytest.mark.parametrize(
    "line_clip, args",
    [
        (cohen_sutherland_line_box_clip, (BOX,)),
        (liang_barsky_line_box_clip, (BOX,)),
        (cyrus_beck_line_polygon_clip, (POLY,)),
        (cyrus_beck_line_polygon_clip, (POLY_REVERSED,)),
    ],
)
def test_line_box_clip_degeneracy(line_clip, args):
    def assert_expected(
        a: tuple,
        b: tuple,
        intersects: bool,
        c: tuple = (np.nan, np.nan),
        d: tuple = (np.nan, np.nan),
    ) -> None:
        a = Point(*a)
        b = Point(*b)
        # c, d are the clipped points
        actual, actual_c, actual_d = line_clip(a, b, *args)
        print(actual_c, c)
        print(actual_d, d)
        assert intersects is actual
        assert np.allclose(actual_c, c, equal_nan=True)
        assert np.allclose(actual_d, d, equal_nan=True)

        # Direction of the point doesn't change, so neither should the answer.
        actual, actual_c, actual_d = line_clip(a, b, *args)
        assert intersects is actual
        assert np.allclose(actual_c, c, equal_nan=True)
        assert np.allclose(actual_d, d, equal_nan=True)

    # Line through vertices
    assert_expected(
        (-1.0, -1.0),
        (3.0, 3.0),
        True,
        (0.0, 0.0),
        (2.0, 2.0),
    )

    # Identity
    assert_expected(
        (-1.0, -1.0),
        (-1.0, -1.0),
        False,
    )

    # Line through lower edge
    assert_expected(
        (-1.0, 0.0),
        (3.0, 0.0),
        True,
        (0.0, 0.0),
        (2.0, 0.0),
    )

    # Line through upper edge
    assert_expected(
        (-1.0, 2.0),
        (3.0, 2.0),
        True,
        (0.0, 2.0),
        (2.0, 2.0),
    )

    # Partial line lower edge
    assert_expected(
        (-1.0, 0.0),
        (1.0, 0.0),
        True,
        (0.0, 0.0),
        (1.0, 0.0),
    )

    # Partial line upper edge
    assert_expected(
        (-1.0, 2.0),
        (1.0, 2.0),
        True,
        (0.0, 2.0),
        (1.0, 2.0),
    )

    # Within lower edge
    assert_expected(
        (0.5, 0.0),
        (1.5, 0.0),
        True,
        (0.5, 0.0),
        (1.5, 0.0),
    )

    # Within upper edge
    assert_expected(
        (0.5, 2.0),
        (1.5, 2.0),
        True,
        (0.5, 2.0),
        (1.5, 2.0),
    )

    # Identical to lower edge
    assert_expected(
        (0.0, 0.0),
        (2.0, 0.0),
        True,
        (0.0, 0.0),
        (2.0, 0.0),
    )

    # Identical to upper edge
    assert_expected(
        (0.0, 2.0),
        (2.0, 2.0),
        True,
        (0.0, 2.0),
        (2.0, 2.0),
    )

    # Identical to left edge
    assert_expected(
        (0.0, 0.0),
        (0.0, 2.0),
        True,
        (0.0, 0.0),
        (0.0, 2.0),
    )

    # Identical to right edge
    assert_expected(
        (2.0, 0.0),
        (2.0, 2.0),
        True,
        (2.0, 0.0),
        (2.0, 2.0),
    )

    # Within left edge
    assert_expected(
        (0.0, 0.5),
        (0.0, 1.5),
        True,
        (0.0, 0.5),
        (0.0, 1.5),
    )

    # Within right edge
    assert_expected(
        (2.0, 0.5),
        (2.0, 1.5),
        True,
        (2.0, 0.5),
        (2.0, 1.5),
    )

    # Identical to left edge
    assert_expected(
        (0.0, 0.0),
        (0.0, 2.0),
        True,
        (0.0, 0.0),
        (0.0, 2.0),
    )

    # Identical to right edge
    assert_expected(
        (2.0, 0.0),
        (2.0, 2.0),
        True,
        (2.0, 0.0),
        (2.0, 2.0),
    )

    # Diagonal line of length (1, 1), touching upper left corner.
    assert_expected(
        (-1.0, 1.0),
        (0.0, 2.0),
        False,
        (np.nan, np.nan),
        (np.nan, np.nan),
    )
