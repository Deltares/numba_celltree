import numba as nb
import numpy as np

from .constants import PARALLEL, CellTreeData, FloatArray, IntArray, IntDType, Point
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
    polygon_length = face.size

    c = False
    for i in range(polygon_length):
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
        current = tree.nodes[node_index]

        # Check if it's a leaf
        if current["child"] == -1:
            for i in range(current["ptr"], current["ptr"] + current["size"]):
                bbox_index = tree.bb_indices[i]
                if point_in_polygon(bbox_index, point, tree.faces, tree.vertices):
                    return bbox_index
            continue

        dim = 1 if current["dim"] else 0
        left = point[dim] <= current["Lmax"]
        right = point[dim] >= current["Rmin"]

        if left and right:
            if (current["Lmax"] - point[dim]) < (point[dim] - current["Rmin"]):
                size = push(stack, current["child"], size)
                size = push(stack, current["child"] + 1, size)
            else:
                size = push(stack, current["child"] + 1, size)
                size = push(stack, current["child"], size)
        elif left:
            size = push(stack, current["child"], size)
        elif right:
            size = push(stack, current["child"] + 1, size)

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


# Bounding box search functions
@nb.njit(inline="always")
def count_bbox(bbox: FloatArray, tree: CellTreeData):
    stack = allocate_stack()
    stack[0] = 0
    count = 0
    size = 1

    while size > 0:
        node_index, size = pop(stack, size)
        current = tree.nodes[node_index]

        # Check if it's a leaf
        if current["child"] == -1:
            count += current["size"]

        dim = 1 if current["dim"] else 0
        left = bbox[dim * 2 + 1] <= current["Lmax"]
        right = bbox[dim * 2] >= current["Rmin"]

        if left and right:
            size = push(stack, current["child"], size)
            size = push(stack, current["child"] + 1, size)
        elif left:
            size = push(stack, current["child"], size)
        elif right:
            size = push(stack, current["child"] + 1, size)

    return count


@nb.njit(inline="always")
def locate_bbox(bbox: FloatArray, tree: CellTreeData, indices: IntArray):
    stack = allocate_stack()
    stack[0] = 0
    count = 0
    size = 1
    count = 0

    while size > 0:
        node_index, size = pop(stack, size)
        current = tree.nodes[node_index]

        # Check if it's a leaf
        if current["child"] == -1:
            for i in range(current["ptr"], current["ptr"] + current["size"]):
                bbox_index = tree.bb_indices[i]
                indices[count] = bbox_index
                count += 1

        dim = 1 if current["dim"] else 0
        left = bbox[dim * 2 + 1] <= current["Lmax"]
        right = bbox[dim * 2] >= current["Rmin"]

        if left and right:
            size = push(stack, current["child"], size)
            size = push(stack, current["child"] + 1, size)
        elif left:
            size = push(stack, current["child"], size)
        elif right:
            size = push(stack, current["child"] + 1, size)

    return


@nb.njit(parallel=PARALLEL)
def locate_bboxes(
    bbox_coords: FloatArray,
    tree: CellTreeData,
):
    n_bbox = bbox_coords.shape[0]
    counts = np.empty(n_bbox + 1, dtype=IntDType)
    counts[0] = 0
    for i in nb.prange(n_bbox):  # pylint: disable=not-an-iterable
        counts[i + 1] = count_bbox(bbox_coords[i], tree)

    # Run a cumulative sum
    total = 0
    for i in range(1, n_bbox + 1):
        total += counts[i]
        counts[i] = total

    ii = np.empty(total, dtype=IntDType)
    jj = np.empty(total, dtype=IntDType)
    for i in nb.prange(n_bbox):  # pylint: disable=not-an-iterable
        start = counts[i]
        end = counts[i + 1]
        ii[start:end] = i
        indices = jj[start:end]
        # locate_bbox_helper(bbox_coords[i], 0, tree, 0, indices)
        locate_bbox(bbox_coords[i], tree, indices)

    return ii, jj
