CellTree2d
==========

.. automodule:: numba_celltree
    :members:
    :imported-members:
    :undoc-members:
    :show-inheritance:

Changelog
=========

[0.4.1 2025-04-15]
------------------

Changed
~~~~~~~

- If tolerances are not provided, they are now estimated by the package. This
  should be a good default for most cases, but can be overridden by providing a
  custom value :meth:`CellTree2d.locate_points` and
  :meth:`EdgeCellTree2d.locate_points`.

[0.4.0 2025-04-10]
------------------

Added
~~~~~

- ``tolerance`` argument to make tolerance configurable in
  :meth:`CellTree2d.locate_points`,
  :meth:`CellTree2d.compute_barycentric_weights`, and
  :meth:`EdgeCellTree2d.locate_points`. This allows for more lenient queries
  when working with datasets with large spatial coordinates. 

Changed
~~~~~~~

- Edge case handling has been improved. Dynamic tolerances are now
  automatically estimated or can be optionally provided for queries to handle
  floating point errors. In previous versions, the size of a cross product was
  compared with a static tolerance value of 1e-9, which made the tolerance
  effectively an area measure, or relative depending on the length of the edge
  rather than the perpendicular distance to the vertex. The current approach
  computes an actual distance, making the tolerance straightforward to
  interpret. The new defaults should result in fewer false positives and false
  negatives.

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

