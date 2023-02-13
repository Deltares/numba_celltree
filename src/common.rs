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
pub struct Vector {
    pub x: f64,
    pub y: f64,
}

impl Vector {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn cross(&self, other: Vector) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn dot(&self, other: Vector) -> f64 {
        self.x * other.x + self.y * other.y
    }
}
