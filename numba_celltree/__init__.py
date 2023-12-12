from importlib.metadata import version as _version

from .celltree import CellTree2d

try:
    __version__ = _version("numba_celltree")
except Exception:
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

__all__ = ("CellTree2d",)
