from typing import Tuple

import numba as nb
import numpy as np

from numba_celltree.algorithms import (
    cohen_sutherland_line_box_clip,
    cyrus_beck_line_polygon_clip,
)
from numba_celltree.constants import (
    PARALLEL,
    BoolArray,
    CellTreeData,
    FloatArray,
    FloatDType,
    IntArray,
    IntDType,
)
from numba_celltree.geometry_utils import (
    Box,
    Point,
    as_box,
    as_point,
    box_contained,
    boxes_intersect,
    copy_vertices_into,
    lines_intersect,
    point_in_polygon_or_on_edge,
    point_on_edge,
    to_point,
    to_vector,
)
from numba_celltree.utils import (
    allocate_polygon,
    allocate_stack,
    allocate_triple_stack,
    grow,
    pop,
    pop_triple,
    push,
    push_triple,
)


@nb.njit(inline="always")
def concatenate_indices(
    indices: IntArray, counts: IntArray
) -> Tuple[IntArray, IntArray]:
    total_size = sum(counts)
    ii = np.empty(total_size, dtype=IntDType)
    jj = np.empty(total_size, dtype=IntDType)
    start = 0
    for i, size in enumerate(counts):
        end = start + size
        ii[start:end] = indices[i][:size, 0]
        jj[start:end] = indices[i][:size, 1]
        start = end
    return ii, jj


# Inlining saves about 15% runtime
@nb.njit(inline="always")
def locate_point(point: Point, tree: CellTreeData):
    stack = allocate_stack()
    polygon_work_array = allocate_polygon()
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
                face = tree.elements[bbox_index]
                # Make sure polygons to test is contiguous (stack allocated) array
                # This saves about 40-50% runtime
                poly = copy_vertices_into(tree.vertices, face, polygon_work_array)
                if point_in_polygon_or_on_edge(point, poly):
                    return bbox_index
            continue

        dim = 1 if node["dim"] else 0
        left = point[dim] <= node["Lmax"]
        right = point[dim] >= node["Rmin"]
        left_child = node["child"]
        right_child = left_child + 1

        if left and right:
            # This heuristic is worthwhile because a point will fall into a
            # single face -- if found, we can stop.
            if (node["Lmax"] - point[dim]) < (point[dim] - node["Rmin"]):
                stack, size = push(stack, left_child, size)
                stack, size = push(stack, right_child, size)
            else:
                stack, size = push(stack, right_child, size)
                stack, size = push(stack, left_child, size)
        elif left:
            stack, size = push(stack, left_child, size)
        elif right:
            stack, size = push(stack, right_child, size)

    return return_value


@nb.njit(parallel=PARALLEL, cache=True)
def locate_points(
    points: FloatArray,
    tree: CellTreeData,
):
    n_points = len(points)
    result = np.empty(n_points, dtype=IntDType)
    for i in nb.prange(n_points):  # pylint: disable=not-an-iterable
        point = as_point(points[i])
        result[i] = locate_point(point, tree)
    return result


# Inlining saves about 15% runtime
@nb.njit(inline="always")
def locate_point_on_edge(point: Point, tree: CellTreeData):
    stack = allocate_stack()
    stack[0] = 0
    return_value = -1
    size = 1
    # TODO: Rename allocate_polygon or assess if we can allocate a smaller array.
    edge_work_array = allocate_polygon()

    while size > 0:
        node_index, size = pop(stack, size)
        node = tree.nodes[node_index]

        # Check if it's a leaf
        if node["child"] == -1:
            for i in range(node["ptr"], node["ptr"] + node["size"]):
                bbox_index = tree.bb_indices[i]
                edge = tree.elements[bbox_index]
                segment = copy_vertices_into(tree.vertices, edge, edge_work_array)
                if point_on_edge(point, segment):
                    return bbox_index
            continue

        dim = 1 if node["dim"] else 0
        left = point[dim] <= node["Lmax"]
        right = point[dim] >= node["Rmin"]
        left_child = node["child"]
        right_child = left_child + 1

        if left and right:
            # This heuristic is worthwhile because a point will fall into a
            # single face -- if found, we can stop.
            if (node["Lmax"] - point[dim]) < (point[dim] - node["Rmin"]):
                stack, size = push(stack, left_child, size)
                stack, size = push(stack, right_child, size)
            else:
                stack, size = push(stack, right_child, size)
                stack, size = push(stack, left_child, size)
        elif left:
            stack, size = push(stack, left_child, size)
        elif right:
            stack, size = push(stack, right_child, size)

    return return_value


@nb.njit(parallel=PARALLEL, cache=True)
def locate_points_on_edge(
    points: FloatArray,
    tree: CellTreeData,
):
    n_points = len(points)
    result = np.empty(n_points, dtype=IntDType)
    for i in nb.prange(n_points):  # pylint: disable=not-an-iterable
        point = as_point(points[i])
        result[i] = locate_point_on_edge(point, tree)
    return result


@nb.njit(inline="always")
def locate_box(
    box: Box, tree: CellTreeData, indices: IntArray, indices_size: int, index: int
) -> Tuple[int, int]:
    """
    Search the tree for a single axis-aligned box.

    Parameters
    ----------
    box: Box named tuple
    tree: CellTreeData
    indices: IntArray
        Array for results. Contains ``index`` of the box we're searching for
        the first column, and the index of the box in the celltree (if any) in
        the second.
    indices_size: int
        Current number of filled in values in ``indices``.
    index: int
        Current index of the box we're searching.
    """
    tree_bbox = as_box(tree.bbox)
    if not boxes_intersect(box, tree_bbox):
        return 0, indices_size
    stack = allocate_stack()
    stack[0] = 0
    size = 1
    count = 0
    capacity = len(indices)

    while size > 0:
        node_index, size = pop(stack, size)
        node = tree.nodes[node_index]
        # Check if it's a leaf
        if node["child"] == -1:
            # Iterate over the bboxes in the leaf
            for i in range(node["ptr"], node["ptr"] + node["size"]):
                bbox_index = tree.bb_indices[i]
                # As a named tuple: saves about 15% runtime
                leaf_box = as_box(tree.bb_coords[bbox_index])
                if boxes_intersect(box, leaf_box):
                    # Exit if we need to re-allocate the array. Exiting instead
                    # of drawing the re-allocation logic in here makes a
                    # significant runtime difference; seems like numba can
                    # optimize this form better.
                    if indices_size >= capacity:
                        return -1, indices_size
                    indices[indices_size, 0] = index
                    indices[indices_size, 1] = bbox_index
                    indices_size += 1
                    count += 1
        else:
            dim = 1 if node["dim"] else 0
            minimum = 2 * dim
            maximum = 2 * dim + 1
            left = box[minimum] <= node["Lmax"]
            right = box[maximum] >= node["Rmin"]
            left_child = node["child"]
            right_child = left_child + 1

            if left and right:
                stack, size = push(stack, left_child, size)
                stack, size = push(stack, right_child, size)
            elif left:
                stack, size = push(stack, left_child, size)
            elif right:
                stack, size = push(stack, right_child, size)

    return count, indices_size


@nb.njit(cache=True)
def locate_boxes_helper(
    box_coords: FloatArray,
    tree: CellTreeData,
    offset: int,
) -> IntArray:
    n_box = len(box_coords)
    # Ensure the initial indices array isn't too small.
    indices = np.empty((max(n_box, 256), 2), dtype=IntDType)
    total_count = 0
    indices_size = 0
    for box_index in range(n_box):
        box = as_box(box_coords[box_index])
        # Re-allocating here is significantly faster than re-allocating inside
        # of ``locate_box``; presumably because that function is kept simpler and
        # numba can optimize better. Unfortunately, that means we have to keep
        # trying until we succeed here; in most cases, success is immediate as
        # the indices array will have enough capacity.
        while True:
            count, indices_size = locate_box(
                box, tree, indices, indices_size, box_index + offset
            )
            if count != -1:
                break
            # Not enough capacity: grow capacity, discard partial work, retry.
            indices_size = total_count
            indices = grow(indices)
        total_count += count
    return indices, total_count


@nb.njit(cache=True, parallel=PARALLEL)
def locate_boxes(box_coords: FloatArray, tree: CellTreeData, n_chunks: int):
    chunks = np.array_split(box_coords, n_chunks)
    offsets = np.zeros(n_chunks, dtype=IntDType)
    for i, chunk in enumerate(chunks[:-1]):
        offsets[i + 1] = offsets[i] + len(chunk)
    # Setup (dummy) typed list for numba to store parallel results.
    indices = [np.empty((0, 2), dtype=IntDType) for _ in range(n_chunks)]
    counts = np.empty(n_chunks, dtype=IntDType)
    for i in nb.prange(n_chunks):
        indices[i], counts[i] = locate_boxes_helper(chunks[i], tree, offsets[i])
    return concatenate_indices(indices, counts)


@nb.njit(inline="always")
def compute_edge_edge_intersect(
    tree: CellTreeData, bbox_index: int, a: Point, b: Point, work_array: np.ndarray
) -> Tuple[bool, Point, Point]:
    tree_edge = tree.elements[bbox_index]
    p = to_point(tree_edge[0])
    q = to_point(tree_edge[1])
    intersects, c = lines_intersect(
        a, b, p, q
    )  # MODIFY LINES_INTERSECT TO RETURN COMPUTED INTERSECTION
    return intersects, c, c


@nb.njit(inline="always")
def compute_edge_face_intersect(
    tree: CellTreeData, bbox_index: int, a: Point, b: Point, work_array: np.ndarray
) -> Tuple[bool, Point, Point]:
    box = as_box(tree.bb_coords[bbox_index])
    intersects, c, d = cohen_sutherland_line_box_clip(a, b, box)
    if intersects:
        polygon = copy_vertices_into(
            tree.vertices, tree.elements[bbox_index], work_array
        )
        intersects, c, d = cyrus_beck_line_polygon_clip(a, b, polygon)
    return intersects, c, d


def make_locate_edges(intersection_function: nb.types.Callable) -> nb.types.Callable:
    # Inlining this function drives compilation time through the roof. It's
    # probably also a rather bad idea, given its complexity: compared to looking
    # for either boxes or points, checking is more much complicated by involving
    # two intersection algorithms.
    @nb.njit(inline="never")
    def locate_edge(
        a: Point,
        b: Point,
        tree: CellTreeData,
        indices: IntArray,
        intersections: FloatArray,
        indices_size: int,
        index: int,
    ):
        # Check if the entire mesh intersects with the line segment at all
        tree_bbox = as_box(tree.bbox)
        tree_intersects, _, _ = cohen_sutherland_line_box_clip(a, b, tree_bbox)
        if not tree_intersects:
            return 0, indices_size

        V = to_vector(a, b)
        stack = allocate_stack()
        polygon_work_array = allocate_polygon()
        stack[0] = 0
        size = 1
        count = 0
        capacity = len(indices)

        while size > 0:
            node_index, size = pop(stack, size)
            node = tree.nodes[node_index]

            # Check if it's a leaf
            if node["child"] == -1:
                for i in range(node["ptr"], node["ptr"] + node["size"]):
                    bbox_index = tree.bb_indices[i]
                    intersects, c, d = intersection_function(
                        tree, bbox_index, a, b, polygon_work_array
                    )
                    if intersects:
                        # If insufficient capacity, exit.
                        if indices_size >= capacity:
                            return -1, indices_size
                        indices[indices_size, 0] = index
                        indices[indices_size, 1] = bbox_index
                        intersections[indices_size, 0, 0] = c.x
                        intersections[indices_size, 0, 1] = c.y
                        intersections[indices_size, 1, 0] = d.x
                        intersections[indices_size, 1, 1] = d.y
                        indices_size += 1
                        count += 1
                continue

            # Note, "x" is a placeholder for x, y here
            # Contrast with t, which is along vector
            node_dim = 1 if node["dim"] else 0
            dx = V[node_dim]
            if dx > 0.0:
                dx_left = node["Lmax"] - a[node_dim]
                dx_right = node["Rmin"] - b[node_dim]
            else:
                dx_left = node["Lmax"] - b[node_dim]
                dx_right = node["Rmin"] - a[node_dim]

            # Check how origin (a) and end (b) are located compared to box edges
            # (Lmax, Rmin). The box should be investigated if:
            # * the origin is left of Lmax (dx_left >= 0)
            # * the end is right of Rmin (dx_right <= 0)
            left = dx_left >= 0.0
            right = dx_right <= 0.0

            # Now find the intersection coordinates. These have to occur within in
            # the bounds of the vector. Note that if the line has no slope in this
            # dim (dx == 0), we cannot compute the intersection, and we have to
            # defer to the child nodes.
            if dx > 0.0:  # TODO: abs(dx) > EPISLON?
                if left:
                    t_left = dx_left / dx
                    left = t_left >= 0.0
                if right:
                    t_right = dx_right / dx
                    right = t_right <= 1.0
            elif dx < 0.0:
                if left:
                    t_left = 1.0 - (dx_left / dx)
                    left = t_left >= 0.0
                if right:
                    t_right = 1.0 - (dx_right / dx)
                    right = t_right <= 1.0
            # else dx == 0.0. In this case there's no info to extract from this
            # node. We'll fully defer to the children.
            left_child = node["child"]
            right_child = left_child + 1

            if left and right:
                stack, size = push(stack, left_child, size)
                stack, size = push(stack, right_child, size)
            elif left:
                stack, size = push(stack, left_child, size)
            elif right:
                stack, size = push(stack, right_child, size)

        return count, indices_size

    @nb.njit(cache=True)
    def locate_edges_helper(
        edge_coords: FloatArray,
        tree: CellTreeData,
        offset: int,
    ) -> IntArray:
        n_edge = len(edge_coords)
        # Ensure the initial indices array isn't too small.
        n = max(n_edge, 256)
        indices = np.empty((n, 2), dtype=IntDType)
        xy = np.empty((n, 2, 2), dtype=FloatDType)

        total_count = 0
        indices_size = 0
        for edge_index in range(n_edge):
            a = as_point(edge_coords[edge_index, 0])
            b = as_point(edge_coords[edge_index, 1])

            while True:
                count, indices_size = locate_edge(
                    a, b, tree, indices, xy, indices_size, edge_index + offset
                )
                if count != -1:
                    break
                # Not enough capacity: grow capacity, discard partial work, retry.
                indices_size = total_count
                indices = grow(indices)
                xy = grow(xy)

            total_count += count

        return indices, xy, total_count

    @nb.njit(cache=True, parallel=PARALLEL)
    def locate_edges(box_coords: FloatArray, tree: CellTreeData, n_chunks: int):
        chunks = np.array_split(box_coords, n_chunks)
        offsets = np.zeros(n_chunks, dtype=IntDType)
        for i, chunk in enumerate(chunks[:-1]):
            offsets[i + 1] = offsets[i] + len(chunk)

        # Setup (dummy) typed lists for numba to store parallel results.
        indices = [np.empty((0, 2), dtype=IntDType) for _ in range(n_chunks)]
        intersections = [np.empty((0, 2, 2), dtype=FloatDType) for _ in range(n_chunks)]
        counts = np.empty(n_chunks, dtype=IntDType)
        for i in nb.prange(n_chunks):
            indices[i], intersections[i], counts[i] = locate_edges_helper(
                chunks[i], tree, offsets[i]
            )

        total_size = sum(counts)
        xy = np.empty((total_size, 2, 2), dtype=FloatDType)
        start = 0
        for i, size in enumerate(counts):
            end = start + size
            xy[start:end] = intersections[i][:size]
            start = end

        ii, jj = concatenate_indices(indices, counts)
        return ii, jj, xy

    return {
        "locate_edges": locate_edges,
        "locate_edges_helper": locate_edges_helper,
        "locate_edge": locate_edge,
    }


# Use the closure to capture the specific intersection function
edge_edge_functions = make_locate_edges(
    intersection_function=compute_edge_edge_intersect
)
edge_face_functions = make_locate_edges(
    intersection_function=compute_edge_face_intersect
)

# Extract the main functions for normal use
locate_edge_edges = edge_edge_functions["locate_edges"]
locate_edge_faces = edge_face_functions["locate_edges"]

# Expose helper functions for testing
locate_edge_edge = edge_edge_functions["locate_edge"]
locate_edge_face = edge_face_functions["locate_edge"]
locate_edge_edge_helper = edge_edge_functions["locate_edges_helper"]
locate_edge_face_helper = edge_face_functions["locate_edges_helper"]


@nb.njit(cache=True)
def collect_node_bounds(tree: CellTreeData) -> FloatArray:
    # Allocate output array.
    # Per row: xmin, xmax, ymin, ymax
    node_bounds = np.empty((len(tree.nodes), 4), dtype=FloatDType)
    # Set bounds of the first node.
    node_bounds[0, 0] = tree.bbox[0]
    node_bounds[0, 1] = tree.bbox[1]
    node_bounds[0, 2] = tree.bbox[2]
    node_bounds[0, 3] = tree.bbox[3]

    # Stack contains: node_index, parent_index, side (right/left)
    ROOT = 0
    RIGHT = 0
    LEFT = 1
    stack = allocate_triple_stack()
    stack[0, :] = (2, ROOT, RIGHT)  # Right child
    stack[1, :] = (1, ROOT, LEFT)  # Left child
    size = 2

    while size > 0:
        # Collect from stacks
        node_index, parent_index, side, size = pop_triple(stack, size)

        parent = tree.nodes[parent_index]
        bbox = node_bounds[parent_index]
        dim = 1 if parent["dim"] else 0

        # Set parent bounding box first.
        # Then place the single new value for the child.
        node_bounds[node_index, 0] = bbox[0]
        node_bounds[node_index, 1] = bbox[1]
        node_bounds[node_index, 2] = bbox[2]
        node_bounds[node_index, 3] = bbox[3]

        if side:
            bound = parent["Lmax"]
        else:
            bound = parent["Rmin"]

        node_bounds[node_index, dim * 2 + side] = bound

        node = tree.nodes[node_index]
        if node["child"] == -1:
            continue

        left_child = node["child"]
        right_child = left_child + 1

        # Right child
        stack, size = push_triple(stack, right_child, node_index, RIGHT, size)
        # Left child
        stack, size = push_triple(stack, left_child, node_index, LEFT, size)

    return node_bounds


@nb.njit(cache=True)
def validate_node_bounds(tree: CellTreeData, node_bounds: FloatArray) -> BoolArray:
    """
    Traverse the tree. Check whether all children are contained in the bounding
    box.

    For the leaf nodes, check whether the bounding boxes are contained.
    """
    node_validity = np.full(len(tree.nodes), False, dtype=np.bool_)
    stack = allocate_stack()
    stack[0] = 0
    size = 1

    while size > 0:
        node_index, size = pop(stack, size)
        bbox = as_box(node_bounds[node_index])
        node = tree.nodes[node_index]

        # Check if it's a leaf:
        if node["child"] == -1:
            valid = True
            for i in range(node["ptr"], node["ptr"] + node["size"]):
                bbox_index = tree.bb_indices[i]
                leaf_box = as_box(tree.bb_coords[bbox_index])
                valid = valid and box_contained(leaf_box, bbox)
            node_validity[node_index] = valid
            continue

        left_child = node["child"]
        right_child = left_child + 1
        left_box = as_box(node_bounds[left_child])
        right_box = as_box(node_bounds[right_child])
        node_validity[node_index] = box_contained(left_box, bbox) and box_contained(
            right_box, bbox
        )

        stack, size = push(stack, right_child, size)
        stack, size = push(stack, left_child, size)

    return node_validity
