use numpy::Element;
use pyo3::prelude::*;
mod constants;
mod creation;
mod query;

use constants::Node;

#[pymodule]
fn numba_celltree_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Warm up dtype cache at import time for early failure and stable overhead.
    let _ = Node::get_dtype_bound(m.py());
    m.add_function(wrap_pyfunction!(query::locate_points, m)?)?;
    m.add_function(wrap_pyfunction!(query::locate_points_serial, m)?)?;
    m.add_function(wrap_pyfunction!(creation::initialize, m)?)?;
    Ok(())
}
