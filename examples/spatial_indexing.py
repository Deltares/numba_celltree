"""
Spatial indexing
================

The goal of a cell tree is to quickly locate cells of an unstructured mesh.
Unstructured meshes are challening in this regard: for a given point, we cannot
simply compute a row and column number as we would for structured data such as
rasters. The most straightforward procedure is checking every single cell,
until we find the cell which contains the point we're looking for. This is
clearly not efficient. 

A cell tree is bounding volume hierarchy (BVH) which may be used as a spatial
index. A spatial index is a data structure to search a spatial object
efficiently, without exhaustively checking every cell. The implementation in
``numba_celltree`` provides four ways to search the tree:

* Locating single points
* Locating bounding boxes
* Locating convex polygons (e.g. cells of another mesh)
* Locating line segments

This example provides a basic introduction to searching a cell tree for each of
these.

We'll start by importing the required packages with matplotlib for plotting.
We will use ``meshzoo`` to create some examples meshes.
"""
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # small examples, avoid JIT overhead

import matplotlib.pyplot as plt
import meshzoo
import numpy as np
from matplotlib.collections import LineCollection
from numba_celltree import CellTree2d, demo

# Let's go easy for a start with a rectangular mesh:
lower_left = (0.0, 0.0)
upper_right = (10.0, 10.0)
vertices, faces = meshzoo.rectangle_quad(lower_left, upper_right, n=10)

# Determine the edges of the cells, and plot them.
node_x, node_y = vertices.transpose()
edges = demo.edges(faces, -1)

fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")

# Locating points
# ---------------
#
# We'll build a cell tree first, then look for some points.
tree = CellTree2d(vertices, faces)
points = np.array(
    [
        [-5.0, 1.0],
        [4.5, 2.5],
        [6.5, 4.5],
    ]
)
i = tree.locate_points(points)
i

# These numbers are the cell numbers in which we can find the points.
#
# A value of -1 means that a point is not located in any cell.
#
# Let's get rid of the -1 values, and take a look which cells have been found.
# We'll color the found cells red, and we'll draw the nodes to compare.
i = i[i != -1]
fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")
demo.plot_edges(node_x, node_y, demo.edges(faces[i], -1), ax, color="blue", linewidth=3)
ax.scatter(*points.transpose())

# Now let's try a more exotic example.
vertices, faces = meshzoo.disk(5, 5)
vertices += 1.0
vertices *= 5.0
node_x, node_y = vertices.transpose()
edges = demo.edges(faces, -1)

fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")

# There are certainly no rows or columns to speak of!
#
# Let's build a new tree, and look for the same points as before.
tree = CellTree2d(vertices, faces)
i = tree.locate_points(points)
i = i[i != -1]

fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")
demo.plot_edges(node_x, node_y, demo.edges(faces[i], -1), ax, color="blue", linewidth=3)
ax.scatter(*points.transpose())

# It should be clear by now that a point may only fall into a single cell. A
# point may also be out of bounds. If a cell falls exactly on an edge, one of the
# two neighbors will be chosen arbitrarily. At any rate, we can always expect
# one answer per cell.
#
# This is not the case for line segments, bounding boxes, or convex polygons: a
# line may intersect multiple cells, and a bounding box or polygon may contain
# multiple cells.
#
# Locating bounding boxes
# -----------------------
#
# A search of N points will yield N answers (cell numbers). A search of N boxes
# may yield M answers. To illustrate, let's look for all the cells inside of
# a box.
box_coords = np.array(
    [
        [4.0, 8.0, 4.0, 6.0],  # xmin, xmax, ymin, ymax
    ]
)
box_i, cell_i = tree.locate_boxes(box_coords)

fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")
demo.plot_edges(
    node_x, node_y, demo.edges(faces[cell_i], -1), ax, color="blue", linewidth=2
)
demo.plot_boxes(box_coords, ax, color="red", linewidth=3)

# We can also search for multiple boxes:
box_coords = np.array(
    [
        [4.0, 8.0, 4.0, 6.0],
        [0.0, 8.0, 8.0, 10.0],
        [10.0, 13.0, 2.0, 8.0],
    ]
)
box_i, cell_i = tree.locate_boxes(box_coords)
box_i, cell_i

# Note that this method returns two arrays of equal length. The second array
# contains the cell numbers, as usual. The first array contains the index of the
# bounding box in which the respective cells fall. Note that there are only two
# number in ``box_i``: there are no cells located in the third box, as we can
# confirm visually:
cells_0 = cell_i[box_i == 0]
cells_1 = cell_i[box_i == 1]
fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")
demo.plot_edges(
    node_x, node_y, demo.edges(faces[cells_0], -1), ax, color="blue", linewidth=2
)
demo.plot_edges(
    node_x, node_y, demo.edges(faces[cells_1], -1), ax, color="green", linewidth=2
)
demo.plot_boxes(box_coords, ax, color="red", linewidth=3)

# Locating cells
# --------------
#
# Similarly, we can look for other cells (convex polygons):
triangle_vertices = np.array(
    [
        [5.0, 3.0],
        [7.0, 3.0],
        [7.0, 5.0],
        [0.0, 6.0],
        [4.0, 4.0],
        [6.0, 10.0],
    ]
)
triangles = np.array(
    [
        [0, 1, 2],
        [3, 4, 5],
    ]
)
tri_x, tri_y = triangle_vertices.transpose()

tri_i, cell_i = tree.locate_faces(triangle_vertices, triangles, -1)
cells_0 = cell_i[tri_i == 0]
cells_1 = cell_i[tri_i == 1]
fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")
demo.plot_edges(
    node_x, node_y, demo.edges(faces[cells_0], -1), ax, color="blue", linewidth=2
)
demo.plot_edges(
    node_x, node_y, demo.edges(faces[cells_1], -1), ax, color="green", linewidth=2
)
demo.plot_edges(tri_x, tri_y, demo.edges(triangles, -1), ax, color="red", linewidth=3)

# We can also compute how large the overlap is:
tri_i, cell_i, area = tree.intersect_faces(triangle_vertices, triangles, -1)

# Let's color the faces of the mesh by their proportion of overlap. Because our
# mesh is triangular, we can represent the triangles as two collections of
# vectors (V, U). Then the area is half of the absolute value of the cross
# product of U and V.
intersection_faces = faces[cell_i]
intersection_vertices = vertices[intersection_faces]
U = intersection_vertices[:, 1] - intersection_vertices[:, 0]
V = intersection_vertices[:, 2] - intersection_vertices[:, 0]
full_area = 0.5 * np.abs(np.cross(V, U))
proportion = area / full_area

fig, ax = plt.subplots()
colored = ax.tripcolor(
    node_x,
    node_y,
    intersection_faces,
    proportion,
)
demo.plot_edges(node_x, node_y, edges, ax, color="black")
demo.plot_edges(tri_x, tri_y, demo.edges(triangles, -1), ax, color="red", linewidth=3)
fig.colorbar(colored)

# Locating lines
# --------------
#
# As a final example, we will compute the intersections with two lines (edges).
# This returns three arrays of equal length:
#
# * the index of the line
# * the index of the cell
# * the location of the intersections
edge_coords = np.array(
    [
        [[0.0, 0.0], [10.0, 10.0]],
        [[0.0, 10.0], [10.0, 0.0]],
    ]
)
edge_i, cell_i, intersections = tree.intersect_edges(edge_coords)
edge_i, cell_i

# To wrap up, we'll color the intersect faces with the length of the
# intersected line segments. We can easily compute the length of each segment
# with the Euclidian norm (Pythagorean distance):
length = np.linalg.norm(intersections[:, 1] - intersections[:, 0], axis=1)

fig, ax = plt.subplots()
demo.plot_edges(node_x, node_y, edges, ax, color="black")
colored = ax.tripcolor(
    node_x,
    node_y,
    faces[cell_i],
    length,
)
ax.add_collection(LineCollection(edge_coords, color="red", linewidth=3))
fig.colorbar(colored)
