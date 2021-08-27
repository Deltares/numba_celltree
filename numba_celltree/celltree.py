from typing import Tuple

import numpy as np

from .algorithms import area_of_intersection, polygons_intersect
from .constants import (
    FILL_VALUE,
    CellTreeData,
    FloatArray,
    FloatDType,
    IntArray,
    IntDType,
)
from .creation import initialize
from .geometry_utils import build_bboxes
from .query import locate_boxes, locate_edges, locate_points


# Ensure all types are as as statically expected.
def cast_vertices(vertices: FloatArray, copy: bool = False) -> FloatArray:
    if isinstance(vertices, np.ndarray):
        vertices = vertices.astype(FloatDType, copy=copy)
    else:
        vertices = np.ascontiguousarray(vertices, dtype=FloatDType)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must be a Nx2 array")
    return vertices


def cast_faces(faces: IntArray, fill_value: int, copy: bool = False) -> IntArray:
    if isinstance(faces, np.ndarray):
        faces = faces.astype(IntDType, copy=copy)
    else:
        faces = np.ascontiguousarray(faces, dtype=IntDType)
    if faces.ndim != 2:
        raise ValueError("faces must be a 2D array")
    if fill_value != FILL_VALUE:
        faces[faces == fill_value] = FILL_VALUE
    return faces


def cast_edges(edges: FloatArray) -> FloatArray:
    edges = np.ascontiguousarray(edges, dtype=FloatDType)
    if edges.ndim != 3 or edges.shape[1] != 2 or edges.shape[2] != 2:
        raise ValueError("edges must be a Nx2x2 array")
    return edges


def bbox_tree(bb_coords: FloatArray) -> FloatArray:
    xmin = bb_coords[:, 0].min()
    xmax = bb_coords[:, 1].max()
    ymin = bb_coords[:, 2].min()
    ymax = bb_coords[:, 3].max()
    return np.array([xmin, xmax, ymin, ymax], dtype=FloatDType)


class CellTree2d:
    def __init__(
        self,
        vertices: FloatArray,
        faces: IntArray,
        n_buckets: int = 4,
        cells_per_leaf: int = 2,
        fill_value=-1,
    ):
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        if cells_per_leaf < 1:
            raise ValueError("cells_per_leaf must be >= 1")

        vertices = cast_vertices(vertices, copy=True)
        faces = cast_faces(faces, fill_value, copy=True)

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
        Parameters
        ----------
        points: (Nx2) FloatArray

        Returns
        -------
        tree_face_indices: IntArray of size N
        """
        points = cast_vertices(points)
        return locate_points(points, self.celltree_data)

    def locate_boxes(self, bbox_coords) -> Tuple[IntArray, IntArray]:
        """
        Parameters
        ----------
        bbox_coords: (Nx4) FloatArray
            Every row containing (xmin, xmax, ymin, ymax)

        Returns
        -------
        bbox_indices: IntArray of size M
            Indices of the bounding box
        tree_face_indices: IntArray of size M
            Indices of the face
        """
        bbox_coords = bbox_coords.astype(FloatDType)
        return locate_boxes(bbox_coords, self.celltree_data)

    def _locate_faces(
        self, vertices: FloatArray, faces: IntArray
    ) -> Tuple[IntArray, IntArray]:
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

    def locate_faces(
        self, vertices: FloatArray, faces: IntArray, fill_value: int
    ) -> Tuple[IntArray, IntArray]:
        """
        Parameters
        ----------
        vertices: Nx2 FloatArrray
        faces: M x n_max_vert IntArray
        fill_value: int

        Returns
        -------
        frace_indices: IntArray of size M
            Indices of the faces
        tree_face_indices: IntArray of size M
            Indices of the tree faces
        """
        vertices = cast_vertices(vertices)
        faces = cast_faces(faces, fill_value)
        return self._locate_faces(vertices, faces)

    def intersect_faces(
        self, vertices: FloatArray, faces: IntArray, fill_value
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        """
        Parameters
        ----------
        vertices: Nx2 FloatArrray
        faces: M x n_max_vert IntArray
        fill_value: int

        Returns
        -------
        frace_indices: IntArray of size M
            Indices of the faces
        tree_face_indices: IntArray of size M
            Indices of the tree faces
        area: FloatArray of size M
            Area of intersection
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
        return i, j, area

    def intersect_edges(
        self, edge_coords: FloatArray
    ) -> Tuple[IntArray, IntArray, FloatArray]:
        edge_coords = cast_edges(edge_coords)
        return locate_edges(edge_coords, self.celltree_data)
