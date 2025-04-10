from numba_celltree.constants import TOLERANCE_FACTOR, MIN_TOLERANCE
from numba_celltree.celltree_base import default_tolerance, bbox_tree, bbox_distances
import numpy as np

bbox_coords = np.array(
    [
        [1.0, 2.0, 1.0, 2.0],
        [4.0, 5.0, 0.0, 1.0],
        [4.0, 5.0, 2.0, 3.0],
        [-1.0, 0.0, 0.0, 4.0],
        [6.0, 8.0, 0.0, 4.0],
        [0.0, 6.0, -1.0, 0.0],
        [0.0, 6.0, 4.0, 5.0],
    ]
)

def test_default_tolerance():
    bb_diagonal = np.array([2.4, 0.5])
    tolerance = default_tolerance(bb_diagonal)
    expected_value = 2.4 * TOLERANCE_FACTOR
    np.testing.assert_allclose(tolerance, expected_value, rtol=0, atol=MIN_TOLERANCE/1e5)

    tolerance = default_tolerance(bb_diagonal/1e4)
    np.testing.assert_allclose(tolerance, MIN_TOLERANCE, rtol=0, atol=MIN_TOLERANCE/1e5)


def test_bbox_tree():
    expected_bbox = np.array([-1.0, 8.0, -1.0, 5.0])
    bbox = bbox_tree(bbox_coords)
    np.testing.assert_allclose(bbox, expected_bbox, rtol=0, atol=1e-5)


def test_bbox_distances():
    expected_distances = np.array(
        [
            [1.0, 1.0, 1.41421356],
            [1.0, 1.0, 1.41421356],
            [1.0, 1.0, 1.41421356],
            [1.0, 4.0, 4.12310563],
            [2.0, 4.0, 4.47213595],
            [6.0, 1.0, 6.08276253],
            [6.0, 1.0, 6.08276253],
        ]
    )
    distances = bbox_distances(bbox_coords)
    np.testing.assert_allclose(distances, expected_distances, rtol=0, atol=1e-5)