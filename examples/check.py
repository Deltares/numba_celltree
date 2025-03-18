# %%

from typing import NamedTuple

# %%

TOLERANCE_ON_EDGE = 1.0e-9


class Point(NamedTuple):
    x: float
    y: float


class Vector(NamedTuple):
    x: float
    y: float


def cross_product(u: Vector, v: Vector) -> float:
    return u.x * v.y - u.y * v.x


def to_vector(a: Point, b: Point) -> Vector:
    return Vector(b.x - a.x, b.y - a.y)


def has_overlap(a: float, b: float, p: float, q: float):
    return ((min(a, b) - max(p, q)) < TOLERANCE_ON_EDGE) and (
        (min(p, q) - max(a, b)) < TOLERANCE_ON_EDGE
    )


def left_of(a: Point, p: Point, U: Vector) -> bool:
    # Whether point a is left of vector U
    # U: p -> q direction vector
    # return U.x * (a.y - p.y + TOLERANCE_ON_EDGE) > U.y * (a.x - p.x - TOLERANCE_ON_EDGE)
    return (U.y * (a.x - p.x) + U.x * (a.y - p.y)) < (4.5 * TOLERANCE_ON_EDGE)


def lines_intersect(a: Point, b: Point, p: Point, q: Point) -> bool:
    """Test whether line segment a -> b intersects p -> q."""
    V = to_vector(a, b)
    U = to_vector(p, q)

    # No intersection if no length.
    if (U.x == 0 and U.y == 0) or (V.x == 0 and V.y == 0):
        return False
    # If x- or y-components are zero, they can only intersect if x or y is identical.
    if (U.x == 0) and (V.x == 0) and a.x != p.x:
        return False
    if (U.y == 0) and (V.y == 0) and a.y != p.y:
        return False

    # bounds check
    if (not has_overlap(a.x, b.x, p.x, q.x)) or (not has_overlap(a.y, b.y, p.y, q.y)):
        return False

    # Check a and b for separation by U (p -> q)
    # and p and q for separation by V (a -> b)
    if (left_of(a, p, U) != left_of(b, p, U)) and (
        left_of(p, a, V) != left_of(q, a, V)
    ):
        return True

    # Detect collinear case, where segments lie on the same infite line.
    R = to_vector(a, p)
    S = to_vector(a, q)
    if (abs(cross_product(V, R)) < TOLERANCE_ON_EDGE) and (
        abs(cross_product(V, S) < TOLERANCE_ON_EDGE)
    ):
        return True

    return False


# %%

a = Point(0.0, 0.0)
b = Point(4.0, 4.0)
# edge not intersection
# p = Point(3.0, 2.0)
# q = Point(3.0, 1.0)
# edge on edge, orthogonal orientation
# p = Point(1.0, 3.0)
# q = Point(3.0, 1.0)
# Vertex on edge
p = Point(-1.0, 1.0)
q = Point(1.0, -1.0)
# Vertex on vertex
# p = Point(0.0, 0.0)
# q = Point(1.0, -1.0)
# Vertex on vertex, equal orientation
# p = Point(0.0, 0.0)
# q = Point(-1.0, -1.0)
# edge on edge, equal orientation
# p = Point(1.0, 1.0)
# q = Point(3.0, 3.0)
U = to_vector(a, b)

# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(*a)
ax.scatter(*b)
ax.plot([a.x, b.x], [a.y, b.y])
ax.scatter(*p)
ax.scatter(*q)
ax.plot([p.x, q.x], [p.y, q.y])
lines_intersect(a, b, p, q)
# %%
