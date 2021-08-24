import numba as nb
import numpy as np

from .constants import (
    FILL_VALUE,
    PARALLEL,
    CellTreeData,
    FloatArray,
    IntArray,
    IntDType,
)
from .geometry_utils import Box, Point, boxes_intersect, polygon_length
from .utils import allocate_stack, pop, push


# Point search functions
@nb.njit(inline="always")
def point_in_polygon(
    bbox_index: int,
    point: Point,
    faces: IntArray,
    vertices: FloatArray,
) -> bool:
    face = faces[bbox_index]
    n_vertex = polygon_length(face)

    c = False
    for i in range(n_vertex):
        v1 = vertices[face[i - 1]]
        v2 = vertices[face[i]]
        # Do not split this in two conditionals: if the first conditional fails,
        # the second will not be executed in Python's (and C's) execution model.
        # This matters because the second can result in division by zero.
        if (v1[1] > point[1]) != (v2[1] > point[1]) and point[0] < (
            (v2[0] - v1[0]) * (point[1] - v1[1]) / (v2[1] - v1[1]) + v1[0]
        ):
            c = not c

    return c


@nb.njit(inline="always")
def locate_point(point: Point, tree: CellTreeData):
    stack = allocate_stack()
    stack[0] = 0
    return_value = -1
    size = 1

    while size > 0:
        node_index, size = pop(stack, size)
        node = tree.nodes[node_index]

        # Check if it's a leaf
        if node["child"] == -1:
            for i in range(node["ptr"], node["ptr"] + node["size"]):
                bbox_index = tree.bb_indices[i]
                if point_in_polygon(bbox_index, point, tree.faces, tree.vertices):
                    return bbox_index
            continue

        dim = 1 if node["dim"] else 0
        left = point[dim] <= node["Lmax"]
        right = point[dim] >= node["Rmin"]

        if left and right:
            if (node["Lmax"] - point[dim]) < (point[dim] - node["Rmin"]):
                size = push(stack, node["child"], size)
                size = push(stack, node["child"] + 1, size)
            else:
                size = push(stack, node["child"] + 1, size)
                size = push(stack, node["child"], size)
        elif left:
            size = push(stack, node["child"], size)
        elif right:
            size = push(stack, node["child"] + 1, size)

    return return_value


@nb.njit(parallel=PARALLEL)
def locate_points(
    points: FloatArray,
    tree: CellTreeData,
):
    n_points = points.shape[0]
    result = np.empty(n_points, dtype=IntDType)
    for i in nb.prange(n_points):  # pylint: disable=not-an-iterable
        point = Point(points[i, 0], points[i, 1])
        result[i] = locate_point(point, tree)
    return result


@nb.njit(inline="always")
def box_from_array(arr: FloatArray) -> Box:
    return Box(
        arr[0],
        arr[1],
        arr[2],
        arr[3],
    )


@nb.njit(inline="always")
def locate_box(box: Box, tree: CellTreeData, indices: IntArray, store_indices: bool):
    if not boxes_intersect(box, tree.bbox):
        return 0
    stack = allocate_stack()
    stack[0] = 0
    size = 1
    count = 0

    while size > 0:
        node_index, size = pop(stack, size)
        print(size)
        node = tree.nodes[node_index]
        # Check if it's a leaf
        if node["child"] == -1:
            # Iterate over the bboxes in the leaf
            for i in range(node["ptr"], node["ptr"] + node["size"]):
                bbox_index = tree.bb_indices[i]
                leaf_box = tree.bb_coords[bbox_index]
                if boxes_intersect(box, leaf_box):
                    if store_indices:
                        indices[count] = bbox_index
                    count += 1
        else:
            dim = 1 if node["dim"] else 0
            minimum = 2 * dim
            maximum = 2 * dim + 1
            left = box[maximum] <= node["Lmax"]
            right = box[minimum] >= node["Rmin"]

            if left and right:
                size = push(stack, node["child"], size)
                size = push(stack, node["child"] + 1, size)
            elif left:
                size = push(stack, node["child"], size)
            elif right:
                size = push(stack, node["child"] + 1, size)

    return count


@nb.njit
def locate_boxes(
    box_coords: FloatArray,
    tree: CellTreeData,
):
    # Numba does not support a concurrent list or bag like stucture:
    # https://github.com/numba/numba/issues/5878
    # (Standard list are not thread safe.)
    # To support parallel execution, we're stuck with numpy arrays therefore.
    # Since we don't know the number of contained bounding boxes, we traverse
    # the tree twice: first to count, then allocate, then another time to
    # actually store the indices.
    # The cost of traversing twice is roughly a factor two. Since many
    # computers can parallellize over more than two threads, counting first --
    # which enables parallelization -- should still result in a net speed up.
    n_box = box_coords.shape[0]
    counts = np.empty(n_box + 1, dtype=IntDType)
    dummy = np.empty((), dtype=IntDType)
    counts[0] = 0
    # First run a count so we can allocate afterwards
    for i in nb.prange(n_box):  # pylint: disable=not-an-iterable
        box = box_from_array(box_coords[i])
        counts[i + 1] = locate_box(box, tree, dummy, False)

    # Run a cumulative sum
    total = 0
    for i in range(1, n_box + 1):
        total += counts[i]
        counts[i] = total

    # Now allocate appropriately
    ii = np.empty(total, dtype=IntDType)
    jj = np.empty(total, dtype=IntDType)
    for i in nb.prange(n_box):  # pylint: disable=not-an-iterable
        start = counts[i]
        end = counts[i + 1]
        ii[start:end] = i
        indices = jj[start:end]
        # locate_bbox_helper(bbox_coords[i], 0, tree, 0, indices)
        box = box_from_array(box_coords[i])
        locate_box(box, tree, indices, True)

    return ii, jj
