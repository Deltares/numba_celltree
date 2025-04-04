import abc

import numpy as np

from numba_celltree.constants import (
    TOLERANCE_ON_EDGE,
    BoolArray,
    FloatArray,
    FloatDType,
    IntArray,
)
from numba_celltree.query import (
    collect_node_bounds,
    validate_node_bounds,
)


def bbox_tree(bb_coords: FloatArray) -> FloatArray:
    xmin = bb_coords[:, 0].min()
    xmax = bb_coords[:, 1].max()
    ymin = bb_coords[:, 2].min()
    ymax = bb_coords[:, 3].max()
    return np.array([xmin, xmax, ymin, ymax], dtype=FloatDType)


class CellTree2dBase(abc.ABC):
    @abc.abstractmethod
    def locate_points(
        self, points: FloatArray, tolerance: float = TOLERANCE_ON_EDGE
    ) -> IntArray:
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
