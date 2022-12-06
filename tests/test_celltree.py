import pathlib
import shutil

import numpy as np
import pytest

from numba_celltree import CellTree2d, demo
from numba_celltree.constants import MAX_N_VERTEX


@pytest.fixture
def datadir(tmpdir, request):
    data = pathlib.Path(__file__).parent / "data"
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
    vertices, faces = demo.generate_disk(5, 5)
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
    expected = [-1, 3, 101]
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
            [36, 37, 33, 34, 35, 84, 32, 89, 87, 90, 39, 20, 97, 99, 98, 101],
            [102, 88, 44, 100, 31, 92, 91, 56, 55, 57, 58, 63, 62, 64, 74, 23],
            [24, 76, 75, 25, 26, 29, 30, 79, 80, 27, 28, 81, 82, 70, 69, 16, 17],
            [68, 14, 15, 65],
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
            [97, 99, 103, 98, 101, 102, 44, 100, 104, 51, 29, 30, 79, 80, 77],
            [27, 28, 81, 82, 78, 16, 17, 93, 95, 31, 92, 94, 96, 11, 8, 91, 14],
            [13, 12, 9, 10, 66, 67, 2, 7, 55, 63, 62, 64, 59],
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    expected_j = np.concatenate(
        [
            [71, 23, 24, 25, 84, 86, 89, 87, 90, 55, 63, 64, 47, 60, 118, 116],
            [117, 54, 105, 41, 40, 106, 103, 98, 101, 88, 100, 16, 31, 92, 94],
            [96, 91, 14, 15, 65],
        ]
    )
    i, j, _ = tree.intersect_edges(edge_coords)
    assert np.array_equal(i, expected_i)
    assert np.array_equal(j, expected_j)


def test_compute_barycentric_weights_triangles():
    tree = CellTree2d(nodes, faces, fill_value)
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    face_indices, weights = tree.compute_barycentric_weights(points)

    expected_indices = np.array([0, 0, 1])
    expected_weights = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.25, 0.5],
            [0.5, 0.25, 0.25],
        ]
    )
    assert np.array_equal(face_indices, expected_indices)
    assert np.allclose(weights, expected_weights)


def test_compute_barycentric_weights_triangle_quads():
    nodes = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [4.0, 0.0],
            [4.0, 4.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2, 3],
            [1, 4, 5, 2],
        ]
    )
    fill_value = -1
    tree = CellTree2d(nodes, faces, fill_value)
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    face_indices, weights = tree.compute_barycentric_weights(points)

    expected_indices = np.array([0, 0, 1])
    expected_weights = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.0, 0.0, 0.5],
        ]
    )
    assert np.array_equal(face_indices, expected_indices)
    assert np.allclose(weights, expected_weights)
