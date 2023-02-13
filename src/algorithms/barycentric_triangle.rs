use crate::common::{DVector, Point, Triangle};
use rayon::prelude::*;

fn compute_weights(triangle: Triangle, p: Point) -> (f64, f64, f64) {
    let ab = triangle.b - triangle.a;
    let ac = triangle.c - triangle.a;
    let ap = p - triangle.a;

    let Aa = ab.cross(ap).abs();
    let Ac = ac.cross(ap).abs();
    let A = ab.cross(ac).abs();

    let inv_denom = 1.0 / A;
    let w = inv_denom * Aa;
    let v = inv_denom * Ac;
    let u = 1.0 - v - w;

    return (u, v, w);
}

fn barycentric_triangle_weights(
    points: &[&[f64]],
    face_indices: &[i64],
    faces: &[&[i64; 3]],
    vertices: &[&[f64; 2]],
) -> Vec<[f64; 3]> {
    points
        .par_iter()
        .zip(face_indices)
        .map(|(point, face_index)| {
            if face_index == &-1 {
                return [0., 0., 0.];
            }
            todo!()
        })
        .collect::<Vec<_>>();
    todo!()
}
