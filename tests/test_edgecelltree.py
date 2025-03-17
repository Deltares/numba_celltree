import numpy as np

from numba_celltree import EdgeCellTree2d
from numba_celltree.constants import TOLERANCE_ON_EDGE, CellTreeData

vertices = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=float)
edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)


def test_init():
    tree = EdgeCellTree2d(vertices, edges)
    assert tree.vertices.shape == (4, 2)
    assert tree.edges.shape == (3, 2)
    assert tree.n_buckets == 4
    assert tree.cells_per_leaf == 2
    assert tree.nodes.shape == (3,)
    assert tree.bb_indices.shape == (3,)
    assert tree.bb_coords.shape == (3, 4)
    assert tree.bbox.shape == (4,)
    assert isinstance(tree.celltree_data, CellTreeData)

    np.testing.assert_array_equal(tree.bb_indices, np.array([0, 1, 2], dtype=np.int32))

    expected_bb_coords = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        tree.bb_coords, expected_bb_coords, atol=TOLERANCE_ON_EDGE
    )
    np.testing.assert_allclose(
        tree.bbox, np.array([0.0, 2.0, 0.0, 1.0]), atol=TOLERANCE_ON_EDGE
    )


def test_locate_points():
    tree = EdgeCellTree2d(vertices, edges)
    points = np.array([[0.5, 0.0], [1.5, 0.0], [2.0, 0.5]], dtype=float)
    tree_edge_indices = tree.locate_points(points)
    np.testing.assert_array_equal(
        tree_edge_indices, np.array([0, 1, 2], dtype=np.int32)
    )

    points = np.array([[0.5, 0.5], [1.5, 0.5], [2.0, 0.5]], dtype=float)
    tree_edge_indices = tree.locate_points(points)
    np.testing.assert_array_equal(
        tree_edge_indices, np.array([-1, -1, 2], dtype=np.int32)
    )
