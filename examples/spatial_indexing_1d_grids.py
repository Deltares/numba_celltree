"""
Spatial indexing of 1D grids
============================

This example demonstrates how to use the `numba_celltree` package to index 1D
grids. The package provides a `EdgeCellTree` class that constructs a cell tree
for 1D grids. The package currently supports the following operations:

* Location points
* Locating line segments

This example provides an introduction to searching a cell tree for each of
these.

We'll start by importing the required packages with matplotlib for plotting.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

os.environ["NUMBA_DISABLE_JIT"] = "1"  # small examples, avoid JIT overhead
from numba_celltree import EdgeCellTree2d, demo  # noqa E402

# %%
# Let's start with creating a simple 1D grid with 4 vertices and 3 edges.

vertices, edges = demo.example_1d_grid()

node_x = vertices.T[0]
node_y = vertices.T[1]

fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")

# %%
# Locating points
# ---------------
#
# We'll build a cell tree first, then look for some points.

tree = EdgeCellTree2d(vertices, edges)
points = np.array(
    [
        [0.25, 1.5],
        [0.75, 1.5],
        [2.0, 1.5],  # This one is outside the grid
    ]
)
i = tree.locate_points(points)
i

# %%
# This returns the indices of the edges in the network on which the points
# are located. Note that the last point is outside the grid, so it returns -1.
# We'll have to filter those out first. Let's plot them:

ii = i[i != -1]

fig, ax = plt.subplots()
ax.scatter(*points.transpose())
demo.plot_edges(node_x, node_y, edges, ax, color="black")
demo.plot_edges(node_x, node_y, edges[ii], ax, color="blue", linewidth=3)

# %%
# Locating line segments
# -----------------------
#
# Let's locate some line segments on the grid. We'll start off with creating
# some line segments.

segments = np.array(
    [
        [[0.0, 1.25], [1.5, 1.5]],
        [[1.5, 1.5], [2.25, 3.5]],
    ]
)

fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")
ax.add_collection(LineCollection(segments, color="gray", linewidth=3))

# %%
# Let's now intersect these line segments with the edges in the network.
segment_index, tree_edge_index, xy_interesection = tree.intersect_edges(segments)
xy_interesection

# %%
# The `intersect_edges` method returns the indices of the segments in the input
# that intersect with the edges in the network. It also returns the indices of
# the edges in the network that intersect with the segments, and the coordinates
# of the intersection points. Let's plot the results:
#

fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")
demo.plot_edges(node_x, node_y, edges[tree_edge_index], ax, color="blue", linewidth=3)
ax.add_collection(LineCollection(segments, color="gray", linewidth=3))
ax.scatter(*xy_interesection.transpose(), s=60, color="darkgreen", zorder=2.5)

# %%
