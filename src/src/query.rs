use ndarray::parallel::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use tinyvec::TinyVec;

use crate::constants::{Point, Vector, MAX_N_VERTEX};
use crate::constants::Node;

#[inline]
fn to_vector(a: Point, b: Point) -> Vector {
    Vector {
        x: b.x - a.x,
        y: b.y - a.y,
    }
}

#[inline]
fn cross_product(u: Vector, v: Vector) -> f64 {
    u.x * v.y - u.y * v.x
}

#[inline]
fn length_squared(v: Vector) -> f64 {
    v.x * v.x + v.y * v.y
}

#[inline]
fn within_perpendicular_distance(uxv: f64, u: Vector, tolerance: f64) -> bool {
    (uxv * uxv) < ((tolerance * length_squared(u)) * tolerance)
}

#[inline]
fn in_bounds(p: Point, a: Point, b: Point) -> bool {
    let xmin = a.x.min(b.x);
    let xmax = a.x.max(b.x);
    let ymin = a.y.min(b.y);
    let ymax = a.y.max(b.y);
    let dx = xmax - xmin;
    let dy = ymax - ymin;
    let use_x_bound = dx.abs() >= dy.abs();
    (use_x_bound && (p.x >= xmin && p.x <= xmax))
        || (!use_x_bound && (p.y >= ymin && p.y <= ymax))
}

#[inline]
fn point_in_polygon_or_on_edge(p: Point, poly: &[Point], tolerance: f64) -> bool {
    let length = poly.len();
    if length < 3 {
        return false;
    }
    let mut v0 = poly[length - 1];
    let mut u = to_vector(p, v0);
    let mut c = false;
    for &v1 in poly.iter() {
        if v1.x == v0.x && v1.y == v0.y {
            continue;
        }
        let v = to_vector(p, v1);
        let a = cross_product(u, v);
        let w = to_vector(v0, v1);
        if within_perpendicular_distance(a, w, tolerance) && in_bounds(p, v0, v1) {
            return true;
        }
        if (v0.y > p.y) != (v1.y > p.y)
            && p.x < ((v1.x - v0.x) * (p.y - v0.y) / (v1.y - v0.y) + v0.x)
        {
            c = !c;
        }
        v0 = v1;
        u = v;
    }
    c
}

fn locate_point(
    p: Point,
    nodes: &[Node],
    bb_indices: &[i64],
    elements: numpy::ndarray::ArrayView2<'_, i64>,
    vertices: numpy::ndarray::ArrayView2<'_, f64>,
    tolerance: f64,
    fill_value: i64,
) -> i64 {
    let mut stack: TinyVec<[usize; 64]> = TinyVec::new();
    let mut poly: TinyVec<[Point; MAX_N_VERTEX]> = TinyVec::new();
    stack.push(0);
    let mut found: i64 = -1;

    while let Some(node_index) = stack.pop() {
        let node = nodes[node_index];
        let child = node.child;
        if child == -1 {
            let ptr = node.ptr;
            let size = node.size;
            for offset in 0..size {
                let bb_index = bb_indices[(ptr + offset) as usize];
                if bb_index < 0 {
                    continue;
                }
                let face = elements.row(bb_index as usize);
                poly.clear();
                for &vidx in face.iter() {
                    if vidx == fill_value {
                        break;
                    }
                    let v = vertices.row(vidx as usize);
                    poly.push(Point { x: v[0], y: v[1] });
                }
                if point_in_polygon_or_on_edge(p, &poly, tolerance) {
                    found = bb_index;
                    break;
                }
            }
            if found != -1 {
                break;
            }
            continue;
        }

        let dim = if node.dim != 0 { 1usize } else { 0usize };
        let lmax = node.lmax;
        let rmin = node.rmin;

        let left = if dim == 0 { p.x <= lmax } else { p.y <= lmax };
        let right = if dim == 0 { p.x >= rmin } else { p.y >= rmin };

        let left_child = child as usize;
        let right_child = (child + 1) as usize;

        if left && right {
            let point_dim = if dim == 0 { p.x } else { p.y };
            if (lmax - point_dim) < (point_dim - rmin) {
                stack.push(left_child);
                stack.push(right_child);
            } else {
                stack.push(right_child);
                stack.push(left_child);
            }
        } else if left {
            stack.push(left_child);
        } else if right {
            stack.push(right_child);
        }
    }

    found
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn locate_points(
    py: Python<'_>,
    points: PyReadonlyArray2<f64>,
    nodes: PyReadonlyArray1<Node>,
    bb_indices: PyReadonlyArray1<i64>,
    elements: PyReadonlyArray2<i64>,
    vertices: PyReadonlyArray2<f64>,
    tolerance: f64,
    fill_value: i64,
) -> PyResult<Py<PyArray1<i64>>> {
    let points = points.as_array();
    let bb_indices = bb_indices.as_slice().expect("bb_indices must be contiguous");
    let elements = elements.as_array();
    let vertices = vertices.as_array();

    let n_points = points.shape()[0];
    let mut result_vec = vec![0_i64; n_points];

    let nodes = nodes.as_slice().expect("nodes must be contiguous");

    py.allow_threads(|| {
        points
            .outer_iter()
            .into_par_iter()
            .zip(result_vec.par_iter_mut())
            .for_each(|(point_row, out)| {
                let p = Point {
                    x: point_row[0],
                    y: point_row[1],
                };
                *out = locate_point(
                    p,
                    nodes,
                    bb_indices,
                    elements,
                    vertices,
                    tolerance,
                    fill_value,
                );
            });
    });

    let result = PyArray1::from_vec_bound(py, result_vec);
    Ok(result.unbind())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn locate_points_serial(
    py: Python<'_>,
    points: PyReadonlyArray2<f64>,
    nodes: PyReadonlyArray1<Node>,
    bb_indices: PyReadonlyArray1<i64>,
    elements: PyReadonlyArray2<i64>,
    vertices: PyReadonlyArray2<f64>,
    tolerance: f64,
    fill_value: i64,
) -> PyResult<Py<PyArray1<i64>>> {
    let points = points.as_array();
    let bb_indices = bb_indices
        .as_slice()
        .expect("bb_indices must be contiguous");
    let elements = elements.as_array();
    let vertices = vertices.as_array();

    let n_points = points.shape()[0];
    let mut result_vec = vec![0_i64; n_points];

    let nodes = nodes.as_slice().expect("nodes must be contiguous");

    for (point_row, out) in points.outer_iter().zip(result_vec.iter_mut()) {
        let p = Point {
            x: point_row[0],
            y: point_row[1],
        };
        *out = locate_point(
            p,
            nodes,
            bb_indices,
            elements,
            vertices,
            tolerance,
            fill_value,
        );
    }

    let result = PyArray1::from_vec_bound(py, result_vec);
    Ok(result.unbind())
}
