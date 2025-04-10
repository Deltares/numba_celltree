from typing import Optional, Tuple

import numba as nb
import numpy as np

from numba_celltree.cast import cast_edges, cast_vertices
from numba_celltree.celltree_base import (
    CellTree2dBase,
    bbox_distances,
    bbox_tree,
    default_tolerance,
)
from numba_celltree.constants import (
    CellTreeData,
    FloatArray,
    IntArray,
)
from numba_celltree.creation import initialize
from numba_celltree.geometry_utils import build_edge_bboxes
from numba_celltree.query import locate_edge_edges, locate_points_on_edge


class EdgeCellTree2d(CellTree2dBase):
    """
    Construct a cell tree from 2D vertices and an edges indexing array.

    Parameters
    ----------
    vertices: ndarray of floats with shape ``(n_point, 2)``
        Corner coordinates (x, y) of the cells.
    edges: ndarray of integers with shape ``(n_edge, 2)``
        Index identifying for every edge the indices of its two nodes.
    n_buckets: int, optional, default: 4
        The number of "buckets" used in tree construction. Must be higher
        or equal to 2. Values over 8 provide diminishing returns.
    cells_per_leaf: int, optional, default: 2
        The number of cells in the leaf nodes of the cell tree. Can be set
        to only 1, but this doubles memory footprint for slightly faster
        lookup. Increase this to reduce memory usage at the cost of lookup
        performance.
    """

    def __init__(
        self,
        vertices: FloatArray,
        edges: IntArray,
        n_buckets: int = 4,
        cells_per_leaf: int = 2,
    ):
        # Determine the tolerance for the bounding boxes. This is mainly
        # relevant when edges are axis-aligned.
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        if cells_per_leaf < 1:
            raise ValueError("cells_per_leaf must be >= 1")

        vertices = cast_vertices(vertices, copy=True)

        bb_coords = build_edge_bboxes(edges, vertices)
        nodes, bb_indices = initialize(edges, bb_coords, n_buckets, cells_per_leaf)
        self.vertices = vertices
        self.edges = edges
        self.n_buckets = n_buckets
        self.cells_per_leaf = cells_per_leaf
        self.nodes = nodes
        self.bb_indices = bb_indices
        self.bb_coords = bb_coords
        self.bbox = bbox_tree(bb_coords)
        self.bb_distances = bbox_distances(bb_coords)
        self.celltree_data = CellTreeData(
            self.edges,
            self.vertices,
            self.nodes,
            self.bb_indices,
            self.bb_coords,
            self.bbox,
            self.cells_per_leaf,
        )

    def locate_points(
        self, points: FloatArray, tolerance: Optional[float] = None
    ) -> IntArray:
        """
        Find the index of a face that contains a point.

        Parameters
        ----------
        points: ndarray of floats with shape ``(n_point, 2)``
            Coordinates of the points to be located.
        tolerance: float, optional
            The tolerance used to determine whether a point is on an edge. If
            the distance from the point to the edge is smaller than this value,
            the point is considered to be on the edge. If None, the method tries
            to estimate an appropriate tolerance by multiplying the maximum
            diagonal of the bounding boxes with 1e-12.


        Returns
        -------
        tree_edge_indices: ndarray of integers with shape ``(n_point,)``
            For every point, the index of the edge it falls on. Points not
            falling on any edge are marked with a value of ``-1``.
        """
        if tolerance is None:
            tolerance = default_tolerance(self.bb_distances[:, 2])
        points = cast_vertices(points)
        return locate_points_on_edge(points, self.celltree_data, tolerance)

    def intersect_edges(
        self,
        edge_coords: FloatArray,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Find the index of an edge intersecting with an edge.

        Parameters
        ----------
        edge_coords: ndarray of floats with shape ``(n_edge, 2, 2)``
            Every row containing ``((x0, y0), (x1, y1))``.

        Returns
        -------
        edge_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the bounding box.
        tree_edge_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the edge.
        intersection: ndarray of floats with shape ``(n_found, 2)``
            Coordinate pair of the intersection.
        """
        edge_coords = cast_edges(edge_coords)
        n_chunks = nb.get_num_threads()
        edge_indices, tree_edge_indices, xy = locate_edge_edges(
            edge_coords, self.celltree_data, n_chunks
        )
        intersection_xy = np.ascontiguousarray(xy[:, 0])
        return edge_indices, tree_edge_indices, intersection_xy
