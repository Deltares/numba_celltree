import numba
from numba.pycc import CC
from numba.core import typing, sigutils
from numba.pycc.compiler import ExportEntry

from .constants import NumbaFloatDType, NumbaIntDType, NumbaCellTreeData
from .creation import initialize
from .query import locate_points, locate_bboxes


def export_function(cc, exported_name, func, sig):
    # Non-decorator form of:
    # https://github.com/numba/numba/blob/e031b70c31c8553521ee89b02db8d2d6e995f9eb/numba/pycc/cc.py#L132
    fn_args, fn_retty = sigutils.normalize_signature(sig)
    sig = typing.signature(fn_retty, *fn_args)
    if exported_name in c._exported_functions:
        raise KeyError("duplicated export symbol %s" % (exported_name))
    entry = ExportEntry(exported_name, sig, func)
    cc._exported_functions[exported_name] = entry


cc = CC("compiled_celltree")

export_function(
    cc=cc,
    exported_name="initialize",
    func=initialize,
    sig=(
        NumbaFloatDType[:, :],  # vertices
        NumbaIntDType[:, :],  # faces
        numba.int32,  # n_buckets
        numba.int32,  # cells_per_leaf
    ),
)

export_function(
    cc=cc,
    exported_name="locate_points",
    func=locate_points,
    sig=(
        NumbaFloatDType[:, :],  # points
        NumbaCellTreeData,  # tree
    ),
)

export_function(
    cc=cc,
    exported_name="locate_bboxes",
    func=locate_bboxes,
    sig=(
        NumbaFloatDType[:, :],  # bbox_coords
        NumbaCellTreeData,  # tree
    ),
)

if __name__ == "__main__":
    cc.compile()
