import matplotlib.pyplot as plt
import numpy as np
import pytest

from numba_celltree import demo


def test_close_polygons():
    faces = np.array(
        [
            [0, 1, 2, -1, -1],
            [0, 1, 2, 3, -1],
            [0, 1, 2, 3, 4],
        ]
    )
    closed = demo.close_polygons(faces, -1)
    expected = np.array(
        [
            [0, 1, 2, 0, 0, 0],
            [0, 1, 2, 3, 0, 0],
            [0, 1, 2, 3, 4, 0],
        ]
    )
    assert np.array_equal(closed, expected)


def test_edges():
    faces = np.array(
        [
            [0, 1, 2, -1],
            [1, 3, 4, 2],
        ]
    )
    actual = demo.edges(faces, -1)
    expected = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 4],
        ]
    )
    assert np.array_equal(actual, expected)


def test_plot_edges():
    _, ax = plt.subplots()
    node_x = np.array([0.0, 1.0, 1.0, 2.0, 2.0])
    node_y = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    edges = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 4],
        ]
    )
    demo.plot_edges(node_x, node_y, edges, ax)


def test_plot_boxes():
    boxes = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
        ]
    )
    _, ax = plt.subplots()
    demo.plot_boxes(boxes, ax)

    boxes = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
        ]
    )
    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        demo.plot_boxes(boxes, ax)


def test_generate_disk():
    with pytest.raises(ValueError, match="partitions should be >= 3"):
        demo.data.generate_disk(2, 2)

    nodes, faces = demo.data.generate_disk(4, 1)
    assert nodes.shape == (5, 2)
    assert faces.shape == (4, 3)
    _, faces = demo.data.generate_disk(4, 2)
    assert faces.shape == (16, 3)
