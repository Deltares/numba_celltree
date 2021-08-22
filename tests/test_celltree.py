import numpy as np
import pytest

from numba_celltree import CellTree2d


@pytest.fixture(scope="function")
def triangle_mesh():
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 0.0])
    fill_value = -1
    # Two triangles
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ]
    )
    vertices = np.column_stack((x, y))
    return vertices, faces, fill_value


def test_init_locate_points(triangle_mesh):
    vertices, faces, _ = triangle_mesh
    tree = CellTree2d(vertices, faces)
    points = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.25],
            [1.5, 0.25],
            [2.5, 0.25],
        ]
    )
    actual = tree.locate_points(points)
    expected = np.array([0, 0, 1, -1])
    assert np.array_equal(actual, expected)
