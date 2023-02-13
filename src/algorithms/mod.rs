mod barycentric_triangle;

use crate::common::{DVector, Point};
use numpy::ndarray::ArrayView2;
use smallvec::SmallVec;

fn inside(p: Point, r: Point, U: DVector) -> bool {
    U.x * (p.y - r.y) > U.y * (p.x - r.x)
}

fn intersection(a: Point, V: DVector, r: Point, N: DVector) -> Option<Point> {
    let W = DVector::new(r.x - a.x, r.y - a.y);
    let nw = N.dot(W);
    let nv = N.dot(V);
    if nv != 0.0 {
        let t = nw / nv;
        Some(Point::new(a.x + t * V.x, a.y + t * V.y))
    } else {
        None
    }
}

fn polygon_area(polygon: impl IntoIterator<Item = Point>) -> f64 {
    let mut area = 0.0;

    let mut iter = polygon.into_iter();

    let a = iter.next().expect("There needs to be a first element.");
    let b = iter.next().expect("There needs to be a second element.");
    let mut U = DVector::new(b.x - a.x, b.y - a.y);
    for c in iter {
        let V = DVector::new(a.x - c.x, a.y - c.y);
        area += U.cross(V).abs();
        U = V;
    }

    0.5 * area
}

pub fn clip_polygons(polygon: ArrayView2<'_, f64>, clipper: ArrayView2<'_, f64>) -> f64 {
    let mut output = SmallVec::<[Point; 6]>::new();
    let mut subject = SmallVec::<[Point; 6]>::new();

    for p in polygon.outer_iter() {
        output.push(Point::new(p[0], p[1]));
    }

    let n_clip = clipper.shape()[0];
    let mut r = Point::new(clipper[[n_clip - 1, 0]], clipper[[n_clip - 1, 1]]);
    for i in 0..n_clip {
        let s = Point::new(clipper[[i, 0]], clipper[[i, 1]]);
        let U = DVector::new(s.x - r.x, s.y - r.y);
        let N = DVector::new(-U.y, U.x);

        if U.x == 0. && U.y == 0. {
            continue;
        }

        subject.clear();
        for &o in &output {
            subject.push(o);
        }
        output.clear();
        let mut a = *subject.last().expect("There needs to be a last element.");
        let mut a_inside = inside(a, r, U);

        for &b in &subject {
            let V = DVector::new(b.x - a.x, b.y - a.y);

            if V.x == 0. && V.y == 0. {
                continue;
            }

            let mut b_inside = inside(b, r, U);

            if b_inside {
                if !a_inside {
                    if let Some(p) = intersection(a, V, r, N) {
                        output.push(p);
                    }
                }
                output.push(b);
            } else if a_inside {
                if let Some(p) = intersection(a, V, r, N) {
                    output.push(p);
                } else {
                    b_inside = true;
                    output.push(b);
                }
            }
            a = b;
            a_inside = b_inside;
        }
        if output.len() < 3 {
            return 0.0;
        }
        r = s;
    }

    return polygon_area(output);
}
