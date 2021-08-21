"""
Types and constants
"""
from typing import NamedTuple
import numba.types as nbtypes
import numba as nb
import numpy as np

IntDType = np.int32
FloatDType = np.float64
IntArray = np.ndarray
FloatArray = np.ndarray
BucketArray = np.ndarray
NodeArray = np.ndarray


class Point(NamedTuple):
    x: FloatDType
    y: FloatDType


class Vector(NamedTuple):
    x: FloatDType
    y: FloatDType


class Node(NamedTuple):
    child: IntDType
    Lmax: FloatDType
    Rmin: FloatDType
    ptr: IntDType
    size: IntDType
    dim: bool


class Bucket(NamedTuple):
    Max: FloatDType
    Min: FloatDType
    Rmin: FloatDType
    Lmax: FloatDType
    index: IntDType
    size: IntDType


class CellTreeData(NamedTuple):
    faces: IntArray
    vertices: FloatArray
    nodes: NodeArray
    bb_indices: IntArray
    cells_per_leaf: int


NodeDType = np.dtype(
    [
        # Index of left child. Right child is child + 1.
        ("child", IntDType),
        # Range of the bounding boxes inside of the node.
        ("Lmax", FloatDType),
        ("Rmin", FloatDType),
        # Index into the bounding box index array, bb_indices.
        ("ptr", IntDType),
        # Number of bounding boxes in this node.
        ("size", IntDType),
        # False = 0 = x, True = 1 = y.
        ("dim", bool),
    ]
)


BucketDType = np.dtype(
    [
        # Range of the bucket.
        ("Max", FloatDType),
        ("Min", FloatDType),
        # Range of the bounding boxes inside the bucket.
        ("Rmin", FloatDType),
        ("Lmax", FloatDType),
        # Index into the bounding box index array, bb_indices.
        ("index", IntDType),
        # Number of bounding boxes in this bucket.
        ("size", IntDType),
    ]
)

PARALLEL = True
# 2D is still rather hard-baked in, so changing this alone to 3 will NOT
# suffice to generalize it to a 3D CellTree.
NDIM = 2

# Derived types & constants
NumbaFloatDType = nb.from_dtype(FloatDType)
NumbaIntDType = nb.from_dtype(IntDType)
NumbaNodeDType = nb.from_dtype(NodeDType)

NumbaCellTreeData = nbtypes.Tuple(
    NumbaIntDType[:, :],  # faces
    NumbaFloatDType[:, :],  # vertices
    NumbaNodeDType[:],  # nodes
    NumbaIntDType[:],  # bb_indices
    nb.int32,  # cells_per_leaf
)

FLOAT_MIN = np.finfo(FloatDType).min
FLOAT_MAX = np.finfo(FloatDType).max
INT_MAX = np.iinfo(IntDType).max
