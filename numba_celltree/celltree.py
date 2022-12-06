from typing import Tuple

import numpy as np

from .algorithms import (
    area_of_intersection,
    barycentric_triangle_weights,
    barycentric_wachspress_weights,
    box_area_of_intersection,
    polygons_intersect,
)
from .constants import (
    FILL_VALUE,
    MAX_N_FACE,
    MAX_N_VERTEX,
    CellTreeData,
    FloatArray,
    FloatDType,
    IntArray,
    IntDType,
)
from .creation import initialize
from .geometry_utils import build_bboxes, counter_clockwise
from .query import locate_boxes, locate_edges, locate_points


# Ensure all types are as as statically expected.
def cast_vertices(vertices: FloatArray, copy: bool = False) -> FloatArray:
    if isinstance(vertices, np.ndarray):
        vertices = vertices.astype(FloatDType, copy=copy)
    else:
        vertices = np.ascontiguousarray(vertices, dtype=FloatDType)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (n_points, 2)")
    return vertices


def cast_faces(faces: IntArray, fill_value: int) -> IntArray:
    if isinstance(faces, np.ndarray):
        faces = faces.astype(IntDType, copy=True)
    else:
        faces = np.ascontiguousarray(faces, dtype=IntDType)
    if faces.ndim != 2:
        raise ValueError("faces must have shape (n_face, n_max_vert)")
    n_face, n_max_vert = faces.shape
    if n_face > MAX_N_FACE:
        raise ValueError(
            f"faces contains {n_face} faces. "
            f"numba_celltree supports a maximum of {MAX_N_FACE} faces. "
            f"Increase MAX_N_FACE in the source code, or supply a smaller mesh."
        )
    if n_max_vert > MAX_N_VERTEX:
        raise ValueError(
            f"faces contains up to {n_max_vert} vertices for a single face. "
            f"numba_celltree supports a maximum of {MAX_N_VERTEX} vertices. "
            f"Increase MAX_N_VERTEX in the source code, or alter the mesh."
        )
    if fill_value != FILL_VALUE:
        faces[faces == fill_value] = FILL_VALUE
    return faces


def cast_bboxes(bbox_coords: FloatArray) -> FloatArray:
    bbox_coords = np.ascontiguousarray(bbox_coords, dtype=FloatDType)
    if bbox_coords.ndim != 2 or bbox_coords.shape[1] != 4:
        raise ValueError("bbox_coords must have shape (n_box, 4)")
    return bbox_coords


def cast_edges(edges: FloatArray) -> FloatArray:
    edges = np.ascontiguousarray(edges, dtype=FloatDType)
    if edges.ndim != 3 or edges.shape[1] != 2 or edges.shape[2] != 2:
        raise ValueError("edges must have shape (n_edge, 2, 2)")
    return edges


def bbox_tree(bb_coords: FloatArray) -> FloatArray:
    xmin = bb_coords[:, 0].min()
    xmax = bb_coords[:, 1].max()
    ymin = bb_coords[:, 2].min()
    ymax = bb_coords[:, 3].max()
    return np.array([xmin, xmax, ymin, ymax], dtype=FloatDType)


class CellTree2d:
    """
    Construct a cell tree from 2D vertices and a faces indexing array.

    Parameters
    ----------
    vertices: ndarray of floats with shape ``(n_point, 2)``
        Corner coordinates (x, y) of the cells.
    faces: ndarray of integers with shape ``(n_face, n_max_vert)``
        Index identifying for every face the indices of its corner nodes.  If a
        face has less corner nodes than ``n_max_vert``, its last indices should
        be equal to ``fill_value``.
    n_buckets: int, optional, default: 4
        The number of "buckets" used in tree construction. Must be higher
        or equal to 2. Values over 8 provide diminishing returns.
    cells_per_leaf: int, optional, default: 2
        The number of cells in the leaf nodes of the cell tree. Can be set
        to only 1, but this doubles memory footprint for slightly faster
        lookup. Increase this to reduce memory usage at the cost of lookup
        performance.
    fill_value: int, optional, default: -1
        Fill value marking empty nodes in ``faces``.
    """

    def __init__(
        self,
        vertices: FloatArray,
        faces: IntArray,
        fill_value: int,
        n_buckets: int = 4,
        cells_per_leaf: int = 2,
    ):
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        if cells_per_leaf < 1:
            raise ValueError("cells_per_leaf must be >= 1")

        vertices = cast_vertices(vertices, copy=True)
        faces = cast_faces(faces, fill_value)
        counter_clockwise(vertices, faces)

        nodes, bb_indices, bb_coords = initialize(
            vertices, faces, n_buckets, cells_per_leaf
        )
        self.vertices = vertices
        self.faces = faces
        self.n_buckets = n_buckets
        self.cells_per_leaf = cells_per_leaf
        self.nodes = nodes
        self.bb_indices = bb_indices
        self.bb_coords = bb_coords
        self.bbox = bbox_tree(bb_coords)
        self.celltree_data = CellTreeData(
            self.faces,
            self.vertices,
            self.nodes,
            self.bb_indices,
            self.bb_coords,
            self.bbox,
            self.cells_per_leaf,
        )

    def locate_points(self, points: FloatArray) -> IntArray:
        """
        Finds the index of a face that contains a point.

        Parameters
        ----------
        points: ndarray of floats with shape ``(n_point, 2)``

        Returns
        -------
        tree_face_indices: ndarray of integers with shape ``(n_point,)``
            For every point, the index of the face it falls in. Points not
            falling in any faces are marked with a value of ``-1``.
        """
        points = cast_vertices(points)
        return locate_points(points, self.celltree_data)

    def locate_boxes(self, bbox_coords: FloatArray) -> Tuple[IntArray, IntArray]:
        """
        Finds the index of a face intersecting with a bounding box.

        Parameters
        ----------
        bbox_coords: ndarray of floats with shape ``(n_box, 4)``
            Every row containing ``(xmin, xmax, ymin, ymax)``.

        Returns
        -------
        bbox_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the bounding box.
        tree_face_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the face.
        """
        bbox_coords = cast_bboxes(bbox_coords)
        return locate_boxes(bbox_coords, self.celltree_data)

    def intersect_boxes(self, bbox_coords: FloatArray) -> Tuple[IntArray, IntArray]:
        """
        Finds the index of a box intersecting with a face, and the area
        of intersection.

        Parameters
        ----------
        bbox_coords: ndarray of floats with shape ``(n_box, 4)``
            Every row containing ``(xmin, xmax, ymin, ymax)``.

        Returns
        -------
        bbox_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the bounding box.
        tree_face_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the tree faces.
        area: ndarray of floats with shape ``(n_found,)``
            Area of intersection between the two intersecting faces.
        """
        bbox_coords = cast_bboxes(bbox_coords)
        i, j = locate_boxes(bbox_coords, self.celltree_data)
        area = box_area_of_intersection(
            bbox_coords=bbox_coords,
            vertices=self.vertices,
            faces=self.faces,
            indices_bbox=i,
            indices_face=j,
        )
        # Separating axes declares polygons with shared edges as touching.
        # Make sure we only include actual intersections.
        actual = area > 0
        return i[actual], j[actual], area[actual]

    def _locate_faces(
        self, vertices: FloatArray, faces: IntArray
    ) -> Tuple[IntArray, IntArray]:
        counter_clockwise(vertices, faces)
        bbox_coords = build_bboxes(faces, vertices)
        shortlist_i, shortlist_j = locate_boxes(bbox_coords, self.celltree_data)
        intersects = polygons_intersect(
            vertices_a=vertices,
            vertices_b=self.vertices,
            faces_a=faces,
            faces_b=self.faces,
            indices_a=shortlist_i,
            indices_b=shortlist_j,
        )
        return shortlist_i[intersects], shortlist_j[intersects]

    def intersect_faces(
        self, vertices: FloatArray, faces: IntArray, fill_value: int
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Finds the index of a face intersecting with another face, and the area
        of intersection.

        Parameters
        ----------
        vertices: ndarray of floats with shape ``(n_point, 2)``
            Corner coordinates (x, y) of the cells.
        faces: ndarray of integers with shape ``(n_face, n_max_vert)``
            Index identifying for every face the indices of its corner nodes.
            If a face has less corner nodes than n_max_vert, its last indices
            should be equal to ``fill_value``.
        fill_value: int, optional, default: -1
            Fill value marking empty nodes in ``faces``.

        Returns
        -------
        frace_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the faces.
        tree_face_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the tree faces.
        area: ndarray of floats with shape ``(n_found,)``
            Area of intersection between the two intersecting faces.
        """
        vertices = cast_vertices(vertices)
        faces = cast_faces(faces, fill_value)
        i, j = self._locate_faces(vertices, faces)
        area = area_of_intersection(
            vertices_a=vertices,
            vertices_b=self.vertices,
            faces_a=faces,
            faces_b=self.faces,
            indices_a=i,
            indices_b=j,
        )
        # Separating axes declares polygons with shared edges as touching.
        # Make sure we only include actual intersections.
        actual = area > 0
        return i[actual], j[actual], area[actual]

    def intersect_edges(
        self, edge_coords: FloatArray
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Finds the index of a face intersecting with an edge.

        Parameters
        ----------
        edge_coords: ndarray of floats with shape ``(n_edge, 2, 2)``
            Every row containing ``((x0, y0), (x1, y1))``.

        Returns
        -------
        edge_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the bounding box.
        tree_face_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the face.
        length: ndarray of floats with shape ``(n_found,)``
            Length of intersection of the edge inside of the face.
        """
        edge_coords = cast_edges(edge_coords)
        return locate_edges(edge_coords, self.celltree_data)

    def compute_barycentric_weights(
        self,
        points: FloatArray,
    ) -> Tuple[IntArray, FloatArray]:
        """
        Computes barycentric weights for points located inside of the grid.

        Parameters
        ----------
        points: ndarray of floats with shape ``(n_point, 2)``

        Returns
        -------
        tree_face_indices: ndarray of integers with shape ``(n_point,)``
            For every point, the index of the face it falls in. Points not
            falling in any faces are marked with a value of ``-1``.
        barycentric_weights: ndarray of integers with shape ``(n_point, n_max_vert)``
            For every point, the barycentric weights of the vertices of the
            face in which the point is located. For points not falling in any
            faces, the weight of all vertices is 0.
        """
        face_indices = self.locate_points(points)
        n_max_vert = self.faces.shape[1]
        if n_max_vert > 3:
            f = barycentric_wachspress_weights
        else:
            f = barycentric_triangle_weights

        weights = f(
            points,
            face_indices,
            self.faces,
            self.vertices,
        )
        return face_indices, weights
