Numba Celltree
==============

Finding your way around in an unstructured meshes is difficult. Numba Celltree
provides methods for searching for points, lines, boxes, and cells (convex
polygons) in a two dimensional unstructured mesh.

.. code:: python

    import meshzoo
    import numpy as np
    from numba_celltree import CellTree2d


    vertices, faces = meshzoo.disk(5, 5)
    vertices += 1.0
    vertices *= 5.0
    tree = CellTree2d(vertices, faces, -1)

    # Intersection with two triangles
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
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    tri_i, cell_i, area = tree.intersect_faces(triangle_vertices, triangles, -1)

    # Intersection with two lines
    edge_coords = np.array(
        [
            [[0.0, 0.0], [10.0, 10.0]],
            [[0.0, 10.0], [10.0, 0.0]],
        ]
    )
    edge_i, cell_i, intersections = tree.intersect_edges(edge_coords)

.. image:: _static/intersection-example.svg

Installation
------------

.. code:: console

   pip install numba_celltree

.. toctree::
   :titlesonly:
   :hidden:

   examples/index
   api
