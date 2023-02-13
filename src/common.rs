use std::ops::Sub;

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn from_array(xy: &[f64]) -> Self {
        Point::new(xy[0], xy[1])
    }
}

impl Sub<Point> for Point {
    type Output = DVector;

    fn sub(self, other: Point) -> DVector {
        DVector::new(self.x - other.x, self.y - other.y)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub a: Point,
    pub b: Point,
    pub c: Point,
}

impl Triangle {
    pub fn new(a: Point, b: Point, c: Point) -> Self {
        Self { a, b, c }
    }

    pub fn from_face(vertices: &[&[f64; 2]], face: &[i64]) -> Self {
        Triangle::new(
            Point::from_array(vertices[face[0] as usize]),
            Point::from_array(vertices[face[1] as usize]),
            Point::from_array(vertices[face[2] as usize]),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DVector {
    pub x: f64,
    pub y: f64,
}

impl DVector {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn cross(&self, other: DVector) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn dot(&self, other: DVector) -> f64 {
        self.x * other.x + self.y * other.y
    }
}
