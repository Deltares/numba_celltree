import numpy as np
import pytest

from numba_celltree import CellTree2d

# some very simple test data:
# two triangles:
nodes2 = [
    [0.0, 0.0],
    [2.0, 0.0],
    [1.0, 2.0],
    [3.0, 2.0],
]

faces2 = [
    [0, 1, 2],
    [1, 3, 2],
]
nodes = np.array(nodes2, dtype=np.float64)
faces = np.array(faces2, dtype=np.intc)


nodes21 = [
    (5, 1),
    (10, 1),
    (3, 3),
    (7, 3),
    (9, 4),
    (12, 4),
    (5, 5),
    (3, 7),
    (5, 7),
    (7, 7),
    (9, 7),
    (11, 7),
    (5, 9),
    (8, 9),
    (11, 9),
    (9, 11),
    (11, 11),
    (7, 13),
    (9, 13),
    (7, 15),
]

faces21 = [
    (0, 1, 3),
    (0, 2, 6),
    (0, 3, 6),
    (1, 3, 4),
    (1, 4, 5),
    (2, 6, 7),
    (6, 7, 8),
    (7, 8, 12),
    (6, 8, 9),
    (8, 9, 12),
    (9, 12, 13),
    (4, 5, 11),
    (4, 10, 11),
    (9, 10, 13),
    (10, 11, 14),
    (10, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (15, 16, 18),
    (15, 17, 18),
    (17, 18, 19),
]


def test_init():
    """
    can a tree be initialized
    """
    # with defaults
    tree = CellTree2d(nodes, faces)
    # with everything specified
    tree = CellTree2d(nodes, faces, n_buckets=2, cells_per_leaf=1)
    # with n_buckets
    tree = CellTree2d(nodes, faces, n_buckets=4)
    # with cells_per_leaf
    tree = CellTree2d(nodes, faces, cells_per_leaf=2)


def test_lists():
    """
    python lists should get converted to numpy arrays
    """
    tree = CellTree2d(nodes2, faces2)


def test_types():
    """
    It should auto-cast the types to the right types for you
    """
    nodes = np.array(nodes2, dtype=np.float32)
    faces = np.array(faces2, dtype=np.int32)
    tree = CellTree2d(nodes, faces)


def test_shape_error():
    nodes = [(1, 2, 3), (3, 4, 5), (4, 5, 6)]
    faces = [
        [0, 1, 2],
        [1, 3, 2],
    ]

    with pytest.raises(ValueError):
        # nodes is wrong shape
        tree = CellTree2d(nodes, faces)
        tree = CellTree2d(nodes2, (2, 3, 4, 5))
        tree = CellTree2d(nodes2, ((2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)))


def test_bounds_errors():
    with pytest.raises(ValueError):
        tree = CellTree2d(nodes, faces, cells_per_leaf=-1)

    with pytest.raises(ValueError):
        tree = CellTree2d(nodes, faces, n_buckets=0)


def test_triangle_lookup():
    tree = CellTree2d(nodes, faces)
    point = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [-1.0, 1.0],
        ]
    )  # in triangle 1
    result = tree.locate_points(point)
    expected = np.array([0, 1, -1])
    assert np.array_equal(result, expected)


def test_poly_lookup():
    # A simple quad grid
    nodes = np.array(
        [
            [0.0, 0.0],  # 0
            [0.0, 2.0],  # 1
            [2.0, 0.0],  # 2
            [2.0, 2.0],  # 3
            [4.0, 0.0],  # 4
            [4.0, 2.0],  # 5
            [6.0, 0.0],  # 6
            [6.0, 2.0],  # 7
            [0.0, 4.0],  # 8
            [2.0, 4.0],  # 9
            [4.0, 4.0],  # 10
            [6.0, 4.0],  # 11
        ]
    )

    # quads
    faces1 = np.array(
        [
            [0, 2, 3, 1],
            [4, 6, 7, 5],
        ],
        dtype=np.intc,
    )
    # Pentas
    faces2 = np.array(
        [
            [0, 8, 9, 5, 2],
            [9, 11, 6, 2, 5],
        ],
        dtype=np.intc,
    )

    tree1 = CellTree2d(nodes, faces1, n_buckets=2, cells_per_leaf=1)
    point = np.array(
        [
            [1.0, 1.0],
            [5.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    result = tree1.locate_points(point)
    expected = np.array([0, 1, -1])
    assert np.array_equal(result, expected)

    tree2 = CellTree2d(nodes, faces2, n_buckets=2, cells_per_leaf=1)
    point = np.array(
        [
            [1.0, 2.0],
            [5.0, 2.0],
            [-1.0, 2.0],
        ]
    )
    result = tree2.locate_points(point)
    expected = np.array([0, 1, -1])
    assert np.array_equal(result, expected)


def test_multi_poly_lookup():
    # A simple quad grid
    nodes = np.array(
        [
            [0.0, 0.0],  # 0
            [0.0, 2.0],  # 1
            [2.0, 0.0],  # 2
            [2.0, 2.0],  # 3
            [4.0, 0.0],  # 4
            [4.0, 2.0],  # 5
            [6.0, 0.0],  # 6
            [6.0, 2.0],  # 7
            [0.0, 4.0],  # 8
            [2.0, 4.0],  # 9
            [4.0, 4.0],  # 10
            [6.0, 4.0],  # 11
        ]
    )

    faces = np.array(
        [[0, 8, 9, 5, 2], [9, 11, 7, 5, -1], [4, 7, 6, -1, -1]], dtype=np.intc
    )
    tree = CellTree2d(nodes, faces, n_buckets=2, cells_per_leaf=1)
    point = np.array(
        [
            [1.0, 1.0],
            [5.0, 1.0],
            [5.0, 3.0],
            [-1.0, 1.0],
        ]
    )
    result = tree.locate_points(point)
    expected = np.array([0, 2, 1, -1])
    assert np.array_equal(result, expected)


def test_multipoint():
    tree = CellTree2d(nodes21, faces21)
    points = [
        (4.2, 3.0),
        (7.7, 13.5),
        (3.4, 7.000000001),
        (7.0, 5.0),  # out of bounds points
        (8.66, 10.99),
        (7.3, 0.74),
        (2.5, 5.5),
        (9.8, 12.3),
    ]
    expected = (1, 20, 7, -1, -1, -1, -1, -1)
    actual = tree.locate_points(points)
    assert np.array_equal(actual, expected)


def test_box_lookup():
    nodes = np.array(
        [
            [0.0, 0.0],  # 0
            [0.0, 2.0],  # 1
            [2.0, 0.0],  # 2
            [2.0, 2.0],  # 3
            [4.0, 0.0],  # 4
            [4.0, 2.0],  # 5
            [6.0, 0.0],  # 6
            [6.0, 2.0],  # 7
            [0.0, 4.0],  # 8
            [2.0, 4.0],  # 9
            [4.0, 4.0],  # 10
            [6.0, 4.0],  # 11
        ]
    )

    faces = np.array(
        [[0, 8, 9, 5, 2], [9, 11, 7, 5, -1], [4, 7, 6, -1, -1]], dtype=np.intc
    )
    tree = CellTree2d(nodes, faces, n_buckets=2, cells_per_leaf=1)
    box_coords = np.array(
        [
            [1.0, 2.0, 1.0, 2.0],  # in face 0
            [4.0, 5.0, 0.0, 1.0],  # in face 2
            [4.0, 5.0, 2.0, 3.0],  # in face 1
            [-1.0, 0.0, 0.0, 4.0],  # out of bounds x
            [6.0, 8.0, 0.0, 4.0],  # out of bounds x
            [0.0, 6.0, -1.0, 0.0],  # out of bounds y
            [0.0, 6.0, 4.0, 5.0],  # out of bounds y
        ]
    )
    actual_i, actual_j = tree.locate_boxes(box_coords)
    expected_i = np.array([0, 1, 2])
    expected_j = np.array([0, 2, 1])
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)
