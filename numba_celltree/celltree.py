from typing import Tuple
from .constants import FloatDType, FloatArray, IntDType, IntArray, CellTreeData
from .creation import initialize
from .query import locate_points, locate_bboxes
from . import compiled_celltree as ct


class CellTree2D:
    def __init__(
        self,
        vertices: FloatArray,
        faces: IntArray,
        n_buckets: int = 4,
        cells_per_leaf: int = 2,
        jit=False,
    ):
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        if cells_per_leaf < 1:
            raise ValueError("cells_per_leaf must be >= 1")
        # Ensure all types are as as statically expected.
        vertices = vertices.astype(FloatDType)
        faces = faces.astype(IntDType)

        if jit:
            self._locate_points = locate_points
            self._locate_bboxes = locate_bboxes
            nodes, bb_indices = initialize(vertices, faces, n_buckets, cells_per_leaf)
        else:
            self._locate_points = ct.locate_points
            self._locate_bboxes = ct.locate_bboxes
            nodes, bb_indices = ct.initialize(
                vertices, faces, n_buckets, cells_per_leaf
            )

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
        points = points.astype(FloatDType)
        return self._locate_points(points, self.celltree_data)

    def locate_bboxes(self, bbox_coords) -> Tuple[IntArray, IntArray]:
        bbox_coords = bbox_coords.astype(FloatDType)
        return self._locate_bboxes(bbox_coords, self.celltree_data)
