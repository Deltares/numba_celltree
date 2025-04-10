CellTree2d
==========

.. automodule:: numba_celltree
    :members:
    :imported-members:
    :undoc-members:
    :show-inheritance:

Changelog
=========

[Unreleased]
------------

Added
~~~~~

- ``tolerance`` argument to make tolerance configurable in
  :meth:`CellTree2d.locate_points`, :meth:`EdgeCellTree2d.locate_points`, and
  :class:`EdgeCellTree2d` constructor. This allows for more lenient queries when
  working with datasets with large spatial coordinates. 

Changed
~~~~~~~

- There have been some changes to check edge cases in the algorithms. Instead of
  comparing areas to a fixed tolerance value (1e-9), we now compare distances to
  a small tolerance value, which is either estimated or set by the user. This
  should improve the accuracy of the algorithms and reduce the number of false
  negatives in edge cases.

[0.3.0 2025-03-25]
------------------

Added
~~~~~

- Add :class:`EdgeCellTree2d` class to support 2D queries on 1D networks and
  linear features.


[0.2.2 2024-10-15]
------------------

Fixed
~~~~~

- :meth:`CellTree2d.intersect_edges` could hang indefinitely due to a while
  loop with faulty logic in
  :func:`numba_celltree.algorithms.cohen_sutherland_line_box_clip`. This issue
  seems to appears when an edge vertex lies exactly on top of a bounding box
  vertex of the celltree. The logic has been updated and the while loop exits
  correctly now.

Changed
~~~~~~~

- The parallellization strategy of :meth:`CellTree2d.locate_boxes`,
  :meth:`CellTree2d.intersect_boxes`, :meth:`CellTree2d.locate_faces`,
  :meth:`CellTree2d.intersect_faces`, and :meth:`CellTree2d.intersect_edges`
  has been changed. Instead of querying twice -- once to count, then
  pre-allocate, then again to store result values -- has been replaced by
  manual chunking of input and dynamic allocation per chunk (thread). This
  should result in a net ~30% performance gain in most cases.

