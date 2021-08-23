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
from .query import locate_bboxes, locate_points

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
        self._locate_bboxes = locate_bboxes

        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        if cells_per_leaf < 1:
            raise ValueError("cells_per_leaf must be >= 1")

        vertices = cast_vertices(vertices)
        faces = cast_faces(faces, fill_value)

        nodes, bb_indices = initialize(vertices, faces, n_buckets, cells_per_leaf)
        self.vertices = vertices
        self.faces = faces
        self.n_buckets = n_buckets
        self.cells_per_leaf = cells_per_leaf
        self.nodes = nodes
        self.bb_indices = bb_indices
        self.celltree_data = CellTreeData(
            self.faces,
            self.vertices,
            self.nodes,
            self.bb_indices,
            self.cells_per_leaf,
        )

    def locate_points(self, points: FloatArray) -> IntArray:
        points = cast_vertices(points)
        return self._locate_points(points, self.celltree_data)

    def locate_bboxes(self, bbox_coords) -> Tuple[IntArray, IntArray]:
        bbox_coords = bbox_coords.astype(FloatDType)
        return self._locate_bboxes(bbox_coords, self.celltree_data)
