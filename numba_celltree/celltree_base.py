import abc
from typing import Optional

import numpy as np

from numba_celltree.constants import (
    MIN_TOLERANCE,
    TOLERANCE_FACTOR,
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


def bbox_distances(bb_coords: FloatArray) -> FloatArray:
    """Compute the dx, dy and dxy distances for the bounding boxes.

    Parameters
    ----------
    bb_coords: np.ndarray of shape (n_nodes, 4)
        The bounding box coordinates of the nodes.

    Returns
    -------
    distances: np.ndarray of shape (n_nodes, 3)
        Respectively the dx, dy and dxy distances for the bounding boxes.
    """
    distances = np.empty((bb_coords.shape[0], 3), dtype=FloatDType)
    # dx
    distances[:, 0] = bb_coords[:, 1] - bb_coords[:, 0]
    # dy
    distances[:, 1] = bb_coords[:, 3] - bb_coords[:, 2]
    # dxy
    distances[:, 2] = np.sqrt(distances[:, 0] ** 2 + distances[:, 1] ** 2)
    return distances


def default_tolerance(bb_diagonal: FloatArray) -> float:
    return max(MIN_TOLERANCE, TOLERANCE_FACTOR * max(bb_diagonal))


class CellTree2dBase(abc.ABC):
    @abc.abstractmethod
    def locate_points(
        self, points: FloatArray, tolerance: Optional[float] = None
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
