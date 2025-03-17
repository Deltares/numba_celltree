# Ensure all types are as as statically expected.
from numba_celltree.constants import (
    FILL_VALUE,
    MAX_N_VERTEX,
    FloatArray,
    FloatDType,
    IntArray,
    IntDType,
)


import numpy as np


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
    _, n_max_vert = faces.shape
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
