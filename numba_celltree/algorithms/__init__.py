from numba_celltree.algorithms.barycentric_triangle import barycentric_triangle_weights
from numba_celltree.algorithms.barycentric_wachspress import (
    barycentric_wachspress_weights,
)
from numba_celltree.algorithms.cohen_sutherland import cohen_sutherland_line_box_clip
from numba_celltree.algorithms.cyrus_beck import cyrus_beck_line_polygon_clip
from numba_celltree.algorithms.liang_barsky import liang_barsky_line_box_clip
from numba_celltree.algorithms.separating_axis import polygons_intersect
from numba_celltree.algorithms.sutherland_hodgman import (
    area_of_intersection,
    box_area_of_intersection,
)

__all__ = (
    "barycentric_triangle_weights",
    "barycentric_wachspress_weights",
    "cohen_sutherland_line_box_clip",
    "cyrus_beck_line_polygon_clip",
    "liang_barsky_line_box_clip",
    "polygons_intersect",
    "area_of_intersection",
    "box_area_of_intersection",
)
