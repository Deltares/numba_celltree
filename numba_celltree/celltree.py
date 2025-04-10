from typing import Optional, Tuple

import numba as nb

from numba_celltree.algorithms import (
    area_of_intersection,
    barycentric_triangle_weights,
    barycentric_wachspress_weights,
    box_area_of_intersection,
    polygons_intersect,
)
from numba_celltree.cast import cast_bboxes, cast_edges, cast_faces, cast_vertices
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
from numba_celltree.geometry_utils import build_face_bboxes, counter_clockwise
from numba_celltree.query import (
    locate_boxes,
    locate_edge_faces,
    locate_points,
)


class CellTree2d(CellTree2dBase):
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

        bb_coords = build_face_bboxes(faces, vertices)
        nodes, bb_indices = initialize(faces, bb_coords, n_buckets, cells_per_leaf)
        self.vertices = vertices
        self.faces = faces
        self.n_buckets = n_buckets
        self.cells_per_leaf = cells_per_leaf
        self.nodes = nodes
        self.bb_indices = bb_indices
        self.bb_coords = bb_coords
        self.bbox = bbox_tree(bb_coords)
        self.bb_distances = bbox_distances(bb_coords)
        self.celltree_data = CellTreeData(
            self.faces,
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

        Points that are very close near an edge of a face will also be
        identified as falling within that face.

        Parameters
        ----------
        points: ndarray of floats with shape ``(n_point, 2)``
            Coordinates of the points to be located.
        tolerance: float, optional, default: 1e-9
            The tolerance used to determine whether a point is on an edge.
            If the distance from the point to the edge is smaller than this
            value, the point is considered to be on the edge.

        Returns
        -------
        tree_face_indices: ndarray of integers with shape ``(n_point,)``
            For every point, the index of the face it falls in. Points not
            falling in any faces are marked with a value of ``-1``.
        """
        if tolerance is None:
            tolerance = default_tolerance(self.bb_distances[:, 2])
        points = cast_vertices(points)
        return locate_points(points, self.celltree_data, tolerance)

    def locate_boxes(self, bbox_coords: FloatArray) -> Tuple[IntArray, IntArray]:
        """
        Find the index of a face intersecting with a bounding box.

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
        n_chunks = nb.get_num_threads()
        return locate_boxes(bbox_coords, self.celltree_data, n_chunks)

    def intersect_boxes(
        self, bbox_coords: FloatArray
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Find the index of a box intersecting with a face, and the area
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
        n_chunks = nb.get_num_threads()
        i, j = locate_boxes(bbox_coords, self.celltree_data, n_chunks)
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

    def locate_faces(
        self, vertices: FloatArray, faces: IntArray
    ) -> Tuple[IntArray, IntArray]:
        """
        Find the index of a face intersecting with another face.

        Only sharing an edge also counts as an intersection, due to the use of
        the separating axis theorem to define intersection. The area of the
        overlap is zero in such a case.

        Parameters
        ----------
        vertices: ndarray of floats with shape ``(n_point, 2)``
            Corner coordinates (x, y) of the cells.
        faces: ndarray of integers with shape ``(n_face, n_max_vert)``
            Index identifying for every face the indices of its corner nodes.
            If a face has less corner nodes than n_max_vert, its last indices
            should be equal to ``fill_value``.

        Returns
        -------
        face_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the faces.
        tree_face_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the tree faces.
        """
        counter_clockwise(vertices, faces)
        bbox_coords = build_face_bboxes(faces, vertices)
        n_chunks = nb.get_num_threads()
        shortlist_i, shortlist_j = locate_boxes(
            bbox_coords, self.celltree_data, n_chunks
        )
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
        Find the index of a face intersecting with another face, and the area
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
        face_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the faces.
        tree_face_indices: ndarray of integers with shape ``(n_found,)``
            Indices of the tree faces.
        area: ndarray of floats with shape ``(n_found,)``
            Area of intersection between the two intersecting faces.
        """
        vertices = cast_vertices(vertices)
        faces = cast_faces(faces, fill_value)
        i, j = self.locate_faces(vertices, faces)
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
        self,
        edge_coords: FloatArray,
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Find the index of a face intersecting with an edge.

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
        intersection_edges: ndarray of floats with shape ``(n_found, 2, 2)``
            The resulting intersected edges, every row containing:
            ``((x0, y0), (x1, y1))``.
            The length of each intersected edge can be computed with:
            ``np.linalg.norm(intersections[:, 1] - intersections[:, 0], axis=1)``.
        """
        edge_coords = cast_edges(edge_coords)
        n_chunks = nb.get_num_threads()
        return locate_edge_faces(edge_coords, self.celltree_data, n_chunks)

    def compute_barycentric_weights(
        self,
        points: FloatArray,
        tolerance: Optional[float] = None,
    ) -> Tuple[IntArray, FloatArray]:
        """
        Compute barycentric weights for points located inside of the grid.

        Parameters
        ----------
        points: ndarray of floats with shape ``(n_point, 2)``
            Coordinates of the points to be located.
        tolerance: float, optional, default: 1e-9
            The tolerance used to determine whether a point is on an edge.
            If the distance from the point to the edge is smaller than this
            value, the point is considered to be on the edge.

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
        if tolerance is None:
            tolerance = default_tolerance(self.bb_distances[:, 2])
        face_indices = self.locate_points(points, tolerance)
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
            tolerance,
        )
        return face_indices, weights
