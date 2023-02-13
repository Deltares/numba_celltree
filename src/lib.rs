#![allow(non_snake_case)]
mod algorithm;
mod common;

use crate::algorithm::clip_polygons;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, ArrayView3};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

/// A Python module implemented in Rust.
#[pymodule]
fn celltree_core(_py: Python, m: &PyModule) -> PyResult<()> {
    fn area_of_intersection(
        polygons: ArrayView3<'_, f64>,
        clippers: ArrayView3<'_, f64>,
    ) -> Array1<f64> {
        polygons
            .outer_iter()
            .zip(clippers.outer_iter())
            .map(|(polygon, clipper)| clip_polygons(polygon, clipper))
            .collect()
    }

    fn area_of_intersection_par(
        polygons: ArrayView3<'_, f64>,
        clippers: ArrayView3<'_, f64>,
    ) -> Array1<f64> {
        polygons
            .outer_iter()
            .into_par_iter()
            .zip(clippers.outer_iter())
            .map(|(polygon, clipper)| clip_polygons(polygon, clipper))
            .collect::<Vec<_>>()
            .into()
    }

    #[pyfn(m)]
    #[pyo3(name = "area_of_intersection")]
    fn area_of_intersection_py<'py>(
        py: Python<'py>,
        polygons: PyReadonlyArray3<'_, f64>,
        clippers: PyReadonlyArray3<'_, f64>,
    ) -> &'py PyArray1<f64> {
        let polygons = polygons.as_array();
        let clippers = clippers.as_array();
        let areas = area_of_intersection(polygons, clippers);
        areas.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "area_of_intersection_par")]
    fn area_of_intersection_par_py<'py>(
        py: Python<'py>,
        polygons: PyReadonlyArray3<'_, f64>,
        clippers: PyReadonlyArray3<'_, f64>,
    ) -> &'py PyArray1<f64> {
        let polygons = polygons.as_array();
        let clippers = clippers.as_array();
        let areas = area_of_intersection_par(polygons, clippers);
        areas.into_pyarray(py)
    }
    Ok(())
}
