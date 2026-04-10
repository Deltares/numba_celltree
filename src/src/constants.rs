use numpy::Element;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

pub const MAX_N_VERTEX: usize = 32;
pub const FLOAT_MIN: f64 = f64::MIN;
pub const FLOAT_MAX: f64 = f64::MAX;
pub const INT_MAX: i64 = i64::MAX;

#[derive(Clone, Copy, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Default)]
pub struct Vector {
    pub x: f64,
    pub y: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Node {
    pub child: i64,
    pub lmax: f64,
    pub rmin: f64,
    pub ptr: i64,
    pub size: i64,
    pub dim: i64,
}

unsafe impl Element for Node {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python<'_>) -> pyo3::Bound<'_, numpy::PyArrayDescr> {
        static NODE_DTYPE: GILOnceCell<pyo3::Py<numpy::PyArrayDescr>> =
            GILOnceCell::new();
        let dtype = NODE_DTYPE.get_or_init(py, || {
            let np = py
                .import_bound("numpy")
                .expect("numpy import failed in get_dtype_bound");
            let int64 = np.getattr("int64").expect("numpy.int64 missing");
            let float64 = np.getattr("float64").expect("numpy.float64 missing");
            let dtype = np
                .getattr("dtype")
                .expect("numpy.dtype missing")
                .call1((
                    vec![
                        ("child", int64.clone()),
                        ("Lmax", float64.clone()),
                        ("Rmin", float64.clone()),
                        ("ptr", int64.clone()),
                        ("size", int64.clone()),
                        ("dim", int64.clone()),
                    ],
                ))
                .expect("numpy.dtype construction failed");
            dtype
                .downcast::<numpy::PyArrayDescr>()
                .expect("dtype is not a PyArrayDescr")
                .clone()
                .unbind()
        });
        dtype.bind(py).clone()
    }
}
