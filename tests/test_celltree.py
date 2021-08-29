import pathlib
import shutil

import meshzoo
import numpy as np
import pytest

from numba_celltree import CellTree2d
from numba_celltree.constants import MAX_N_VERTEX


@pytest.fixture
def datadir(tmpdir, request):
    data = pathlib.Path(__file__).parent / "data"
    print(data)
    shutil.copy(data / "triangles.txt", tmpdir / "triangles.txt")
    shutil.copy(data / "xy.txt", tmpdir / "xy.txt")
    return tmpdir


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
fill_value = -1
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
    CellTree2d(nodes, faces, fill_value)
    # with everything specified
    CellTree2d(nodes, faces, fill_value, n_buckets=2, cells_per_leaf=1)
    # with n_buckets
    CellTree2d(nodes, faces, fill_value, n_buckets=4)
    # with cells_per_leaf
    CellTree2d(nodes, faces, fill_value, cells_per_leaf=2)


def test_init_larger_mesh(datadir):
    # This mesh is large enough so that a bucket will get split during
    # construction.
    nodes = np.loadtxt(datadir / "xy.txt", dtype=float)
    faces = np.loadtxt(datadir / "triangles.txt", dtype=int)
    CellTree2d(nodes, faces, fill_value, n_buckets=2)


def test_lists():
    """
    python lists should get converted to numpy arrays
    """
    CellTree2d(nodes2, faces2, fill_value)


def test_types():
    """
    It should auto-cast the types to the right types for you
    """
    nodes = np.array(nodes2, dtype=np.float32)
    faces = np.array(faces2, dtype=np.int32)
    CellTree2d(nodes, faces, fill_value)


def test_fill_value_conversion():
    faces = np.array([[0, 1, 2, -999], [1, 3, 2, -999]])
    tree = CellTree2d(nodes, faces, -999)
    assert tree.faces[0, -1] == -1
    assert tree.faces[1, -1] == -1


def test_shape_errors():
    faces = [0, 1, 2, 1, 3, 2]
    nodes = [(1, 2, 3), (3, 4, 5), (4, 5, 6)]
    box_coords = np.array([0.0, 1.0, 2.0, 3.0])
    edge_coords = np.array([[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        CellTree2d(nodes, faces2, -1)
    with pytest.raises(ValueError):
        CellTree2d(nodes2, faces, -1)

    tree = CellTree2d(nodes2, faces2, -1)
    with pytest.raises(ValueError):
        tree.locate_points(nodes)
    with pytest.raises(ValueError):
        tree.intersect_faces(nodes, faces2, -1)
    with pytest.raises(ValueError):
        tree.intersect_faces(faces, nodes2, -1)
    with pytest.raises(ValueError):
        tree.locate_boxes(box_coords)
    with pytest.raises(ValueError):
        tree.intersect_boxes(box_coords)
    with pytest.raises(ValueError):
        tree.intersect_edges(edge_coords)

    # Can't realistically test MAX_N_FACE: 2e9 faces requires enormous
    # allocation.
    faces = np.arange(MAX_N_VERTEX + 1).reshape((1, -1))
    with pytest.raises(ValueError):
        tree.intersect_faces(nodes2, faces, -1)


def test_bounds_errors():
    with pytest.raises(ValueError):
        CellTree2d(nodes, faces, fill_value, cells_per_leaf=-1)

    with pytest.raises(ValueError):
        CellTree2d(nodes, faces, fill_value, n_buckets=0)


def test_triangle_lookup():
    tree = CellTree2d(nodes, faces, fill_value)
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

    tree1 = CellTree2d(nodes, faces1, fill_value, n_buckets=2, cells_per_leaf=1)
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

    tree2 = CellTree2d(nodes, faces2, fill_value, n_buckets=2, cells_per_leaf=1)
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
    tree = CellTree2d(nodes, faces, fill_value, n_buckets=2, cells_per_leaf=1)
    point = np.array(
        [
            [1.0, 1.0],
            [5.0, 0.5],
            [5.0, 3.0],
            [-1.0, 1.0],
        ]
    )
    result = tree.locate_points(point)
    expected = np.array([0, 2, 1, -1])
    assert np.array_equal(result, expected)


def test_multipoint():
    tree = CellTree2d(nodes21, faces21, fill_value)
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
    tree = CellTree2d(nodes, faces, fill_value, n_buckets=2, cells_per_leaf=1)
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

    actual_i, actual_j, _ = tree.intersect_boxes(box_coords)
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)


def test_edge_lookup():
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
    tree = CellTree2d(nodes, faces, fill_value, n_buckets=2, cells_per_leaf=1)
    edge_coords = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0]],  # 0
            [[4.0, 3.0], [5.0, 4.0]],  # 1
            [[5.0, 0.0], [6.0, 1.0]],  # 2
            [[-2.0, -1.0], [0.0, 1.0]],  # out of bounds
            [[-2.0, -1.0], [-2.0, -1.0]],  # out of bbox
        ]
    )
    actual_i, actual_j, intersections = tree.intersect_edges(edge_coords)
    expected_i = np.array([0, 1, 2])
    expected_j = np.array([0, 1, 2])
    expected_intersections = edge_coords[:3]
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)
    assert np.array_equal(intersections, expected_intersections)

    # Flip edge orientation
    actual_i, actual_j, intersections = tree.intersect_edges(edge_coords[:, ::-1])
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)
    assert np.array_equal(intersections, expected_intersections[:, ::-1])


def test_example_material():
    # Note: the concatenation of lists to get 1D arrays is purely to keep black
    # from formatting everything into very long 1-element columns.
    vertices, faces = meshzoo.disk(5, 5)
    vertices += 1.0
    vertices *= 5.0

    tree = CellTree2d(vertices, faces, -1)
    points = np.array(
        [
            [-5.0, 1.0],
            [4.5, 2.5],
            [6.5, 4.5],
        ]
    )
    expected = [-1, 56, 80]
    assert np.array_equal(tree.locate_points(points), expected)

    box_coords = np.array(
        [
            [4.0, 8.0, 4.0, 6.0],
            [0.0, 8.0, 8.0, 10.0],
            [10.0, 13.0, 2.0, 8.0],
        ]
    )
    expected_i = np.concatenate(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    expected_j = np.concatenate(
        [
            [107, 103, 94, 96, 91, 102, 101, 106, 100, 105, 92, 89, 85, 88, 80],
            [84, 81, 76, 50, 75, 55, 59, 0, 5, 9, 51, 25, 30, 26, 34, 118, 111],
            [115, 120, 122, 114, 117, 123, 124, 119, 121, 4, 8, 7, 3, 15, 12],
            [14, 11, 18, 17, 20, 22],
        ]
    )
    i, j = tree.locate_boxes(box_coords)
    assert np.array_equal(i, expected_i)
    assert np.array_equal(j, expected_j)

    triangle_vertices = np.array(
        [
            [5.0, 3.0],
            [7.0, 3.0],
            [7.0, 5.0],
            [0.0, 6.0],
            [4.0, 4.0],
            [6.0, 10.0],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ]
    )

    expected_i = np.concatenate(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1],
        ]
    )
    expected_j = np.concatenate(
        [
            [85, 88, 80, 84, 77, 81, 76, 59, 66, 63, 123, 124, 116, 119, 121, 4],
            [8, 7, 3, 2, 14, 11, 1, 6, 0, 5, 10, 13, 9, 27, 16, 17, 21, 19, 32],
            [28, 23, 24, 33, 29, 25, 30, 26, 34, 31],
        ]
    )

    i, j, _ = tree.intersect_faces(triangle_vertices, triangles, -1)
    assert np.array_equal(i, expected_i)
    assert np.array_equal(j, expected_j)

    edge_coords = np.array(
        [
            [[0.0, 0.0], [10.0, 10.0]],
            [[0.0, 10.0], [10.0, 0.0]],
        ]
    )
    expected_i = np.concatenate(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    expected_j = np.concatenate(
        [
            [112, 111, 115, 114, 110, 101, 106, 100, 105, 50, 75, 0, 25, 30, 34],
            [41, 38, 46, 44, 48, 49, 100, 83, 78, 82, 79, 80, 77, 81, 76, 50],
            [75, 14, 0, 5, 10, 13, 9, 17, 20, 22, 25],
        ]
    )
    i, j, _ = tree.intersect_edges(edge_coords)
    assert np.array_equal(i, expected_i)
    assert np.array_equal(j, expected_j)
