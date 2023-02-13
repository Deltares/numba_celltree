#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
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
