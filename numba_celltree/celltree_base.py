import abc

import numpy as np

from numba_celltree.constants import (
    BoolArray,
    FloatArray,
    FloatDType,
    IntArray,
)
from numba_celltree.query import (
    collect_node_bounds,
    validate_node_bounds,
)


class CellTree2dBase(abc.ABC):
    @abc.abstractmethod
    def build_bboxes(self, elements: IntArray, vertices: FloatArray) -> FloatArray:
        pass

    @abc.abstractmethod
    def locate_points(self, points: FloatArray) -> IntArray:
        pass

    @property
    def node_bounds(self):
        """Return the bounds (xmin, xmax, ymin, ymax) for every node of the tree."""
        return collect_node_bounds(self.celltree_data)

    def validate_node_bounds(self) -> BoolArray:
        """
        Traverse the tree. Check whether all children are contained in the bounding
        box.

        For the leaf nodes, check whether the bounding boxes are contained.

        Returns
        -------
        node_validity: np.array of bool
            For each node, whether all children are fully contained by its
            bounds.
        """
        return validate_node_bounds(self.celltree_data, self.node_bounds)

    def to_dict_of_lists(self):
        """
        Convert the tree structure to a dict of lists.

        Such a dict can be ingested by e.g. NetworkX to produce visualize the
        tree structure.

        Returns
        -------
        dict_of_lists: Dict[Int, List[Int]]
            Contains for every node a list with its children.

        Examples
        --------
        >>> import networkx
        >>> from networkx.drawing.nx_pydot import graphviz_layout
        >>> d = celltree.to_dict_of_lists()
        >>> G = networkx.DiGraph(d)
        >>> positions = graphviz_layout(G, prog="dot")
        >>> networkx.draw(G, positions, with_labels=True)

        Note that computing the graphviz layout may be quite slow!
        """
        dict_of_lists = {}
        for parent_index, node in enumerate(self.celltree_data.nodes):
            left_child = node["child"]
            if left_child == -1:
                dict_of_lists[parent_index] = []
            else:
                right_child = left_child + 1
                dict_of_lists[parent_index] = [left_child, right_child]

        return dict_of_lists

    @staticmethod
    # Ensure all types are as as statically expected.
    def cast_vertices(vertices: FloatArray, copy: bool = False) -> FloatArray:
        if isinstance(vertices, np.ndarray):
            vertices = vertices.astype(FloatDType, copy=copy)
        else:
            vertices = np.ascontiguousarray(vertices, dtype=FloatDType)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (n_points, 2)")
        return vertices

    @staticmethod
    def cast_bboxes(bbox_coords: FloatArray) -> FloatArray:
        bbox_coords = np.ascontiguousarray(bbox_coords, dtype=FloatDType)
        if bbox_coords.ndim != 2 or bbox_coords.shape[1] != 4:
            raise ValueError("bbox_coords must have shape (n_box, 4)")
        return bbox_coords

    @staticmethod
    def cast_edges(edges: FloatArray) -> FloatArray:
        edges = np.ascontiguousarray(edges, dtype=FloatDType)
        if edges.ndim != 3 or edges.shape[1] != 2 or edges.shape[2] != 2:
            raise ValueError("edges must have shape (n_edge, 2, 2)")
        return edges

    @staticmethod
    def bbox_tree(bb_coords: FloatArray) -> FloatArray:
        xmin = bb_coords[:, 0].min()
        xmax = bb_coords[:, 1].max()
        ymin = bb_coords[:, 2].min()
        ymax = bb_coords[:, 3].max()
        return np.array([xmin, xmax, ymin, ymax], dtype=FloatDType)
