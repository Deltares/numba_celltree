import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
import pathlib
import shutil

import numpy as np
import pytest

from numba_celltree import CellTree2d, demo


@pytest.fixture
def datadir(tmpdir, request):
    data = pathlib.Path(__file__).parent / "data"
    shutil.copy(data / "triangles.txt", tmpdir / "triangles.txt")
    shutil.copy(data / "xy.txt", tmpdir / "xy.txt")
    shutil.copy(data / "voronoi.txt", tmpdir / "voronoi.txt")
    shutil.copy(data / "voronoi_xy.txt", tmpdir / "voronoi_xy.txt")
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


def disk():
    vertices, faces = demo.generate_disk(5, 5)
    centroids = vertices[faces].mean(axis=1)
    rsquared = (centroids[:, 0] - -1.0) ** 2 + (centroids[:, 1] - -1.0) ** 2
    order = np.argsort(rsquared)
    faces = faces[order]
    return vertices, faces


def test_init():
    """Can a tree be initialized?"""
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
    """Python lists should get converted to numpy arrays"""
    CellTree2d(nodes2, faces2, fill_value)


def test_types():
    """It should auto-cast the types to the right types for you"""
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
    assert np.allclose(intersections, expected_intersections)

    # Flip edge orientation
    actual_i, actual_j, intersections = tree.intersect_edges(edge_coords[:, ::-1])
    assert np.array_equal(actual_i, expected_i)
    assert np.array_equal(actual_j, expected_j)
    assert np.allclose(intersections, expected_intersections[:, ::-1])


def test_example_material():
    # Note: the concatenation of lists to get 1D arrays is purely to keep black
    # from formatting everything into very long 1-element columns.
    vertices, faces = disk()
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
    expected = [-1, 24, 63]
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
            [107, 109, 99, 106, 95, 93, 92, 80, 62, 73, 86, 97, 81, 79, 65, 63],
            [76, 57, 44, 59, 54, 58, 50, 45, 43, 31, 41, 36, 38, 27, 123, 112],
            [120, 117, 121, 108, 110, 115, 116, 101, 104, 103, 111, 88, 98],
            [90, 100, 72, 84, 85, 66, 75, 69],
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
            [81, 79, 61, 65, 63, 76, 44, 59, 48, 40, 115, 116, 101],
            [104, 91, 103, 111, 88, 98, 83, 72, 84, 68, 74, 54, 58, 55],
            [67, 47, 35, 50, 66, 46, 53, 26, 34, 49, 56, 29, 37, 43],
            [36, 38, 27, 30],
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
            [124, 112, 120, 108, 93, 96, 80, 62, 73, 43, 36, 27, 16],
            [21, 11, 2, 6, 0, 70, 71, 77, 64, 61, 65, 63, 57, 59],
            [72, 54, 58, 55, 67, 50, 66, 75, 69],
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

    expected_indices = np.array([0, 0, 0])
    expected_weights = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.5, 0.5, 0.0],
        ]
    )
    assert np.array_equal(face_indices, expected_indices)
    assert np.allclose(weights, expected_weights)


def test_node_bounds(datadir):
    nodes = np.loadtxt(datadir / "voronoi_xy.txt", dtype=float)
    faces = np.loadtxt(datadir / "voronoi.txt", dtype=int)
    tree = CellTree2d(nodes, faces, fill_value)
    node_bounds = tree.node_bounds

    n_node = len(tree.celltree_data.nodes)
    xmin, xmax, ymin, ymax = tree.bbox
    assert node_bounds.shape == (n_node, 4)
    assert (node_bounds[:, 0] >= xmin).all()
    assert (node_bounds[:, 1] <= xmax).all()
    assert (node_bounds[:, 2] >= ymin).all()
    assert (node_bounds[:, 3] <= ymax).all()


def test_node_validity(datadir):
    nodes = np.loadtxt(datadir / "voronoi_xy.txt", dtype=float)
    faces = np.loadtxt(datadir / "voronoi.txt", dtype=int)
    tree = CellTree2d(nodes, faces, fill_value)

    assert tree.validate_node_bounds().all()

    # Introduce a failure as described in:
    # https://github.com/Deltares/numba_celltree/issues/2

    tree.nodes[71]["Lmax"] = -0.02319655
    validity = tree.validate_node_bounds()
    assert not validity[73]


def test_find_centroids(datadir):
    nodes = np.loadtxt(datadir / "xy.txt", dtype=float)
    faces = np.loadtxt(datadir / "triangles.txt", dtype=int)
    centroids = nodes[faces].mean(axis=1)
    tree = CellTree2d(nodes, faces, fill_value)
    expected = np.arange(len(centroids))
    actual = tree.locate_points(centroids)
    assert np.array_equal(expected, actual)


def test_to_dict_of_lists(datadir):
    nodes = np.loadtxt(datadir / "xy.txt", dtype=float)
    faces = np.loadtxt(datadir / "triangles.txt", dtype=int)
    tree = CellTree2d(nodes, faces, fill_value, n_buckets=4)
    d = tree.to_dict_of_lists()

    assert isinstance(d, dict)
    assert list(d.keys()) == list(range(len(tree.celltree_data.nodes)))
    assert max(len(v) for v in d.values()) == 2


def test_locate_point_on_edge():
    nodes = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [3.0, 2.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [2, 4, 3],
        ]
    )
    fill_value = -1
    tree = CellTree2d(nodes, faces, fill_value, n_buckets=4)
    points = np.array(
        [
            [0.0, 0.0],
            [0.01, 0.01],
            [0.05, 0.05],
            [0.15, 0.15],
            [0.25, 0.25],
            [0.35, 0.35],
            [0.45, 0.45],
            [0.55, 0.55],
            [0.65, 0.65],
            [0.75, 0.75],
        ]
    )
    result = tree.locate_points(points)
    assert (result != -1).all()

    nodes = np.array(
        [
            [0.0, 0.0],  # 0
            [2.0, 0.0],  # 1
            [2.0, 2.0],  # 2
            [0.0, 2.0],  # 3
            [4.0, 0.0],  # 4
            [4.0, 4.0],  # 5
            [0.0, 4.0],  # 6
        ]
    )
    faces = np.array(
        [
            [0, 1, 2, 3],
            [1, 4, 5, 2],
            [3, 2, 5, 6],
        ]
    )
    fill_value = -1
    tree = CellTree2d(nodes, faces, fill_value, n_buckets=4)
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, 4.0],
            [2.0, 2.0],
            [4.0, 0.0],
            [4.0, 4.0],
        ]
    )
    assert (result != -1).all()
