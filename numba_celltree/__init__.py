from importlib.metadata import version as _version

from numba_celltree.celltree import CellTree2d
from numba_celltree.edge_celltree import EdgeCellTree2d

try:
    __version__ = _version("numba_celltree")
except Exception:
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

__all__ = ("CellTree2d", "EdgeCellTree2d")
