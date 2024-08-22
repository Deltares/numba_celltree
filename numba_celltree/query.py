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
    point_in_polygon_or_on_edge,
    to_vector,
)
from numba_celltree.utils import allocate_polygon, allocate_stack, pop, push


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
                face = tree.faces[bbox_index]
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
                size = push(stack, left_child, size)
                size = push(stack, right_child, size)
            else:
                size = push(stack, right_child, size)
                size = push(stack, left_child, size)
        elif left:
            size = push(stack, left_child, size)
        elif right:
            size = push(stack, right_child, size)

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


@nb.njit(inline="always")
def locate_box(box: Box, tree: CellTreeData, indices: IntArray, store_indices: bool):
    tree_bbox = as_box(tree.bbox)
    if not boxes_intersect(box, tree_bbox):
        return 0
    stack = allocate_stack()
    stack[0] = 0
    size = 1
    count = 0

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
                    if store_indices:
                        indices[count] = bbox_index
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
                size = push(stack, left_child, size)
                size = push(stack, right_child, size)
            elif left:
                size = push(stack, left_child, size)
            elif right:
                size = push(stack, right_child, size)

    return count


@nb.njit(parallel=PARALLEL, cache=True)
def locate_boxes(
    box_coords: FloatArray,
    tree: CellTreeData,
):
    # Numba does not support a concurrent list or bag like stucture:
    # https://github.com/numba/numba/issues/5878
    # (Standard lists are not thread safe.)
    # To support parallel execution, we're stuck with numpy arrays therefore.
    # Since we don't know the number of contained bounding boxes, we traverse
    # the tree twice: first to count, then allocate, then another time to
    # actually store the indices.
    # The cost of traversing twice is roughly a factor two. Since many
    # computers can parallellize over more than two threads, counting first --
    # which enables parallelization -- should still result in a net speed up.
    n_box = box_coords.shape[0]
    counts = np.empty(n_box + 1, dtype=IntDType)
    dummy = np.empty((0,), dtype=IntDType)
    counts[0] = 0
    # First run a count so we can allocate afterwards
    for i in nb.prange(n_box):  # pylint: disable=not-an-iterable
        box = as_box(box_coords[i])
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
        box = as_box(box_coords[i])
        locate_box(box, tree, indices, True)

    return ii, jj


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
    store_intersection: bool,
):
    # Check if the entire mesh intersects with the line segment at all
    tree_bbox = as_box(tree.bbox)
    tree_intersects, _, _ = cohen_sutherland_line_box_clip(a, b, tree_bbox)
    if not tree_intersects:
        return 0

    V = to_vector(a, b)
    stack = allocate_stack()
    polygon_work_array = allocate_polygon()
    stack[0] = 0
    size = 1
    count = 0

    while size > 0:
        node_index, size = pop(stack, size)
        node = tree.nodes[node_index]

        # Check if it's a leaf
        if node["child"] == -1:
            for i in range(node["ptr"], node["ptr"] + node["size"]):
                bbox_index = tree.bb_indices[i]
                box = as_box(tree.bb_coords[bbox_index])
                box_intersect, _, _ = cohen_sutherland_line_box_clip(a, b, box)
                if box_intersect:
                    polygon = copy_vertices_into(
                        tree.vertices, tree.faces[bbox_index], polygon_work_array
                    )
                    face_intersects, c, d = cyrus_beck_line_polygon_clip(a, b, polygon)
                    if face_intersects:
                        if store_intersection:
                            indices[count] = bbox_index
                            intersections[count, 0, 0] = c.x
                            intersections[count, 0, 1] = c.y
                            intersections[count, 1, 0] = d.x
                            intersections[count, 1, 1] = d.y
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
            size = push(stack, left_child, size)
            size = push(stack, right_child, size)
        elif left:
            size = push(stack, left_child, size)
        elif right:
            size = push(stack, right_child, size)

    return count


@nb.njit(parallel=PARALLEL, cache=True)
def locate_edges(
    edge_coords: FloatArray,
    tree: CellTreeData,
):
    # Numba does not support a concurrent list or bag like stucture:
    # https://github.com/numba/numba/issues/5878
    # (Standard lists are not thread safe.)
    # To support parallel execution, we're stuck with numpy arrays therefore.
    # Since we don't know the number of contained bounding boxes, we traverse
    # the tree twice: first to count, then allocate, then another time to
    # actually store the indices.
    # The cost of traversing twice is roughly a factor two. Since many
    # computers can parallellize over more than two threads, counting first --
    # which enables parallelization -- should still result in a net speed up.
    n_edge = edge_coords.shape[0]
    counts = np.empty(n_edge + 1, dtype=IntDType)
    int_dummy = np.empty((0,), dtype=IntDType)
    float_dummy = np.empty((0, 0, 0), dtype=FloatDType)
    counts[0] = 0
    # First run a count so we can allocate afterwards
    for i in nb.prange(n_edge):  # pylint: disable=not-an-iterable
        a = as_point(edge_coords[i, 0])
        b = as_point(edge_coords[i, 1])
        counts[i + 1] = locate_edge(a, b, tree, int_dummy, float_dummy, False)

    # Run a cumulative sum
    total = 0
    for i in range(1, n_edge + 1):
        total += counts[i]
        counts[i] = total

    # Now allocate appropriately
    ii = np.empty(total, dtype=IntDType)
    jj = np.empty(total, dtype=IntDType)
    xy = np.empty((total, 2, 2), dtype=FloatDType)
    for i in nb.prange(n_edge):  # pylint: disable=not-an-iterable
        start = counts[i]
        end = counts[i + 1]
        ii[start:end] = i
        indices = jj[start:end]
        intersections = xy[start:end]
        a = as_point(edge_coords[i, 0])
        b = as_point(edge_coords[i, 1])
        locate_edge(a, b, tree, indices, intersections, True)

    return ii, jj, xy


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

    stack = allocate_stack()
    parent_stack = allocate_stack()
    side_stack = allocate_stack()

    # Right child
    stack[0] = 2
    parent_stack[0] = 0
    side_stack[0] = 0
    # Left child
    stack[1] = 1
    parent_stack[1] = 0
    side_stack[1] = 1
    # Stack size starts at two.
    size = 2

    while size > 0:
        # Collect from stacks
        # Sizes are synchronized.
        parent_index, _ = pop(parent_stack, size)
        side, _ = pop(side_stack, size)
        node_index, size = pop(stack, size)

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
        push(parent_stack, node_index, size)
        push(side_stack, 0, size)
        size = push(stack, right_child, size)

        # Left child
        push(parent_stack, node_index, size)
        push(side_stack, 1, size)
        size = push(stack, left_child, size)

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

        size = push(stack, right_child, size)
        size = push(stack, left_child, size)

    return node_validity
