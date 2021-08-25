from typing import Tuple

import numpy as np

from .constants import (
    FILL_VALUE,
    CellTreeData,
    FloatArray,
    FloatDType,
    IntArray,
    IntDType,
)
from .creation import initialize
from .geometry_utils import Box, build_bboxes
from .query import locate_boxes, locate_points
from .separating_axis import polygons_intersect
from .sutherland_hodgman import area_of_intersection

try:
    from . import aot_compiled
except ImportError:
    import warnings

    warnings.warn(
        "Could not import ahead-of-time compiled celltree. Reinstall package "
        "or use jit=True."
    )


def cast_vertices(vertices: FloatArray) -> FloatArray:
    # Ensure all types are as as statically expected.
    vertices = np.ascontiguousarray(vertices, dtype=FloatDType)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must be a Nx2 array")
    return vertices


def cast_faces(faces: IntArray, fill_value) -> IntArray:
    faces = np.ascontiguousarray(faces, dtype=IntDType)
    if faces.ndim != 2:
        raise ValueError("faces must be a 2D array")
    if fill_value != FILL_VALUE:
        faces[faces == fill_value] = FILL_VALUE
    return faces


def bbox_tree(bb_coords: FloatArray) -> FloatArray:
    xmin = bb_coords[:, 0].min()
    xmax = bb_coords[:, 1].max()
    ymin = bb_coords[:, 2].min()
    ymax = bb_coords[:, 3].max()
    return Box(xmin, xmax, ymin, ymax)


class CellTree2d:
    def __init__(
        self,
        vertices: FloatArray,
        faces: IntArray,
        n_buckets: int = 4,
        cells_per_leaf: int = 2,
        fill_value=-1,
        jit=False,
    ):
        if jit:
            self._initialize = initialize
        else:
            self._initialize = aot_compiled.initialize
        # Compiling query only takes around 5 seconds and is much faster
        # afterwards
        self._locate_points = locate_points
        self._locate_boxes = locate_boxes

        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        if cells_per_leaf < 1:
            raise ValueError("cells_per_leaf must be >= 1")

        vertices = cast_vertices(vertices)
        faces = cast_faces(faces, fill_value)

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
        return self._locate_points(points, self.celltree_data)

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
        return self._locate_boxes(bbox_coords, self.celltree_data)

    def _locate_faces(
        self, vertices: FloatArray, faces: IntArray
    ) -> Tuple[IntArray, IntArray]:
        bbox_coords = build_bboxes(faces, vertices)
        shortlist_i, shortlist_j = self._locate_bboxes(bbox_coords, self.celltree_data)
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
