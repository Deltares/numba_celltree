from numba.core import sigutils, typing
from numba.pycc import CC
from numba.pycc.compiler import ExportEntry

from .constants import NumbaCellTreeData, NumbaFloatDType, NumbaIntDType
from .creation import initialize
from .query import locate_bboxes, locate_points


def export_function(cc, exported_name, func, sig):
    # Non-decorator form of:
    # https://github.com/numba/numba/blob/e031b70c31c8553521ee89b02db8d2d6e995f9eb/numba/pycc/cc.py#L132
    fn_args, fn_retty = sigutils.normalize_signature(sig)
    sig = typing.signature(fn_retty, *fn_args)
    if exported_name in cc._exported_functions:
        raise KeyError("duplicated export symbol %s" % (exported_name))
    entry = ExportEntry(exported_name, sig, func)
    cc._exported_functions[exported_name] = entry


cc = CC("aot_compiled")
# cc.verbose = True

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

export_function(
    cc=cc,
    exported_name="initialize",
    func=initialize,
    sig=(
        NumbaFloatDType[:, :],  # vertices
        NumbaIntDType[:, :],  # faces
        NumbaIntDType,  # n_buckets
        NumbaIntDType,  # cells_per_leaf
    ),
)

if __name__ == "__main__":
    cc.compile()
