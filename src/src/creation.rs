use ndarray::ArrayView2;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::constants::{Node, FLOAT_MAX, FLOAT_MIN, INT_MAX};

#[derive(Clone, Copy)]
struct Bucket {
    max: f64,
    min: f64,
    rmin: f64,
    lmax: f64,
    index: i64,
    size: i64,
}

impl Bucket {
    fn new(max: f64, min: f64, rmin: f64, lmax: f64, index: i64, size: i64) -> Self {
        Self {
            max,
            min,
            rmin,
            lmax,
            index,
            size,
        }
    }
}

#[inline]
fn create_node(ptr: i64, size: i64, dim: i64) -> Node {
    Node {
        child: -1,
        lmax: -1.0,
        rmin: -1.0,
        ptr,
        size,
        dim,
    }
}

#[inline]
fn push_node(nodes: &mut [Node], node: Node, index: usize) -> usize {
    nodes[index] = node;
    index + 1
}

fn stable_partition(
    bb_indices: &mut [i64],
    bb_coords: ArrayView2<'_, f64>,
    begin: usize,
    end: usize,
    bucket: &Bucket,
    dim: usize,
    temp: &mut [i64],
) -> usize {
    let temp = &mut temp[..end - begin];
    let mut count_true = 0usize;
    let mut count_false: isize = -1;

    for &index in bb_indices[begin..end].iter() {
        let bbox = bb_coords.row(index as usize);
        let base = 2 * dim;
        let centroid = bbox[base] + 0.5 * (bbox[base + 1] - bbox[base]);
        if (centroid >= bucket.min) && (centroid < bucket.max) {
            temp[count_true] = index;
            count_true += 1;
        } else {
            let pos = (temp.len() as isize + count_false) as usize;
            temp[pos] = index;
            count_false -= 1;
        }
    }

    for i in 0..count_true {
        bb_indices[begin + i] = temp[i];
    }

    let start_second = begin + count_true;
    let n_false = (-1 - count_false) as usize;
    for i in 0..n_false {
        bb_indices[start_second + i] = temp[temp.len() - 1 - i];
    }

    start_second
}

fn sort_bbox_indices(
    bb_indices: &mut [i64],
    bb_coords: ArrayView2<'_, f64>,
    buckets: &mut Vec<Bucket>,
    node: Node,
    dim: usize,
) {
    let mut current = node.ptr as usize;
    let end = (node.ptr + node.size) as usize;
    let mut temp = vec![0_i64; end - current];

    let b = buckets[0];
    buckets[0] = Bucket::new(b.max, b.min, b.rmin, b.lmax, node.ptr, b.size);

    let mut i = 1usize;
    while current != end {
        let bucket = buckets[i - 1];
        current = stable_partition(
            bb_indices,
            bb_coords,
            current,
            end,
            &bucket,
            dim,
            &mut temp,
        );
        let start = bucket.index as usize;

        let b = buckets[i - 1];
        buckets[i - 1] = Bucket::new(
            b.max,
            b.min,
            b.rmin,
            b.lmax,
            b.index,
            (current - start) as i64,
        );

        if i < buckets.len() {
            let b = buckets[i];
            buckets[i] = Bucket::new(
                b.max,
                b.min,
                b.rmin,
                b.lmax,
                buckets[i - 1].index + buckets[i - 1].size,
                b.size,
            );
        }

        i += 1;
    }
}

fn get_bounds(
    index: i64,
    size: i64,
    bb_coords: ArrayView2<'_, f64>,
    bb_indices: &[i64],
    dim: usize,
) -> (f64, f64) {
    let mut rmin = FLOAT_MAX;
    let mut lmax = FLOAT_MIN;
    let start = index as usize;
    let end = start + size as usize;
    let base = 2 * dim;
    for i in start..end {
        let data_index = bb_indices[i] as usize;
        let bbox = bb_coords.row(data_index);
        let value = bbox[base];
        if value < rmin {
            rmin = value;
        }
        let value = bbox[base + 1];
        if value > lmax {
            lmax = value;
        }
    }
    (rmin, lmax)
}

fn split_plane(
    buckets: &[Bucket],
    root: Node,
    range_lmax: f64,
    range_rmin: f64,
    bucket_length: f64,
) -> (usize, f64, f64) {
    let mut plane_min_cost = FLOAT_MAX;
    let mut plane = INT_MAX;
    let mut bbs_in_left: i64 = 0;

    for i in 1..buckets.len() {
        let current_bucket = buckets[i - 1];
        let next_bucket = buckets[i];
        bbs_in_left += current_bucket.size;
        let bbs_in_right = root.size - bbs_in_left;
        let left_volume = (current_bucket.lmax - range_rmin) / bucket_length;
        let right_volume = (range_lmax - next_bucket.rmin) / bucket_length;
        let plane_cost = left_volume * bbs_in_left as f64 + right_volume * bbs_in_right as f64;
        if plane_cost < plane_min_cost {
            plane_min_cost = plane_cost;
            plane = i as i64;
        }
    }

    let plane = if plane == INT_MAX { 1usize } else { plane as usize };
    let mut lmax = FLOAT_MIN;
    let mut rmin = FLOAT_MAX;
    for i in 0..plane {
        let b_lmax = buckets[i].lmax;
        if b_lmax > lmax {
            lmax = b_lmax;
        }
    }
    for i in plane..buckets.len() {
        let b_rmin = buckets[i].rmin;
        if b_rmin < rmin {
            rmin = b_rmin;
        }
    }

    (plane, lmax, rmin)
}

fn pessimistic_n_nodes(n_elements: usize) -> usize {
    let mut n_nodes = n_elements;
    let mut nodes = ((n_elements as f64) / 2.0).ceil() as usize;
    while nodes > 1 {
        n_nodes += nodes;
        nodes = ((nodes as f64) / 2.0).ceil() as usize;
    }
    n_nodes + 1
}

fn build(
    nodes: &mut [Node],
    mut node_index: usize,
    bb_indices: &mut [i64],
    bb_coords: ArrayView2<'_, f64>,
    n_buckets: usize,
    cells_per_leaf: usize,
) -> usize {
    let mut stack: Vec<(usize, i64)> = Vec::new();
    stack.push((0, 0));

    while let Some((root_index, mut dim)) = stack.pop() {
        let mut dim_flag = dim;
        if dim < 0 {
            dim += 2;
        }

        let root = nodes[root_index];

        if root.size <= cells_per_leaf as i64 {
            continue;
        }

        let dim_usize = dim as usize;
        let (range_rmin, range_lmax) =
            get_bounds(root.ptr, root.size, bb_coords, bb_indices, dim_usize);
        let bucket_length = (range_lmax - range_rmin) / n_buckets as f64;

        let mut buckets = Vec::with_capacity(n_buckets);
        for i in 0..n_buckets {
            buckets.push(Bucket::new(
                (i as f64 + 1.0) * bucket_length + range_rmin,
                i as f64 * bucket_length + range_rmin,
                -1.0,
                -1.0,
                -1,
                0,
            ));
        }

        sort_bbox_indices(bb_indices, bb_coords, &mut buckets, root, dim_usize);

        for i in 0..n_buckets {
            let (rmin, lmax) = get_bounds(
                buckets[i].index,
                buckets[i].size,
                bb_coords,
                bb_indices,
                dim_usize,
            );
            let b = buckets[i];
            buckets[i] = Bucket::new(b.max, b.min, rmin, lmax, b.index, b.size);
        }

        if (cells_per_leaf == 1) && (root.size == 2) {
            nodes[root_index].lmax = range_lmax;
            nodes[root_index].rmin = range_rmin;
            let next_dim = if dim == 0 { 1 } else { 0 };
            let left_child = create_node(root.ptr, 1, next_dim);
            let right_child = create_node(root.ptr + 1, 1, next_dim);
            nodes[root_index].child = node_index as i64;
            node_index = push_node(nodes, left_child, node_index);
            node_index = push_node(nodes, right_child, node_index);
            continue;
        }

        while !buckets.is_empty() && buckets[0].size == 0 {
            let b = buckets[1];
            buckets[1] = Bucket::new(b.max, buckets[0].min, b.rmin, b.lmax, b.index, b.size);
            buckets.remove(0);
        }

        let mut i = 1usize;
        while i < buckets.len() {
            let next_bucket = buckets[i];
            if next_bucket.size == 0 {
                let b = buckets[i - 1];
                buckets[i - 1] = Bucket::new(next_bucket.max, b.min, b.rmin, b.lmax, b.index, b.size);
                buckets.remove(i);
            } else {
                i += 1;
            }
        }

        let mut needs_continue = false;
        for bucket in buckets.iter() {
            if bucket.size == root.size {
                needs_continue = true;
                if dim_flag >= 0 {
                    let next_dim = if dim == 0 { 1 } else { 0 };
                    dim_flag = next_dim - 2;
                    nodes[root_index].dim = next_dim;
                    stack.push((root_index, dim_flag));
                } else {
                    nodes[root_index].lmax = -1.0;
                    nodes[root_index].rmin = -1.0;
                }
                break;
            }
        }
        if needs_continue {
            continue;
        }

        let (plane, lmax, rmin) =
            split_plane(&buckets, root, range_lmax, range_rmin, bucket_length);
        let right_index = buckets[plane].index;
        let right_size = root.ptr + root.size - right_index;
        let left_index = root.ptr;
        let left_size = root.size - right_size;
        nodes[root_index].lmax = lmax;
        nodes[root_index].rmin = rmin;
        let next_dim = if dim == 0 { 1 } else { 0 };
        let left_child = create_node(left_index, left_size, next_dim);
        let right_child = create_node(right_index, right_size, next_dim);
        nodes[root_index].child = node_index as i64;
        let child_index = node_index;
        node_index = push_node(nodes, left_child, node_index);
        node_index = push_node(nodes, right_child, node_index);

        stack.push((child_index + 1, right_child.dim));
        stack.push((child_index, left_child.dim));
    }

    node_index
}

#[pyfunction]
pub fn initialize(
    py: Python<'_>,
    elements: PyReadonlyArray2<i64>,
    bb_coords: PyReadonlyArray2<f64>,
    n_buckets: usize,
    cells_per_leaf: usize,
) -> PyResult<(Py<PyArray1<Node>>, Py<PyArray1<i64>>)> {
    let elements = elements.as_array();
    let bb_coords = bb_coords.as_array();
    let n_elements = elements.shape()[0];
    let mut bb_indices: Vec<i64> = (0..n_elements as i64).collect();

    let n_nodes = pessimistic_n_nodes(n_elements);
    let mut nodes = vec![Node::default(); n_nodes];

    let root = create_node(0, bb_indices.len() as i64, 0);
    let mut node_index = push_node(&mut nodes, root, 0);

    node_index = build(
        &mut nodes,
        node_index,
        &mut bb_indices,
        bb_coords,
        n_buckets,
        cells_per_leaf,
    );

    nodes.truncate(node_index);
    let nodes_array = PyArray1::from_vec_bound(py, nodes);
    let bb_indices_array = PyArray1::from_vec_bound(py, bb_indices);
    Ok((nodes_array.unbind(), bb_indices_array.unbind()))
}
