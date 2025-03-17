"""
Test data generated with:

```python
import numpy as np
import shapely.geometry as sg


def ccw(a):
    # Ensure triangles are counter-clockwise
    for i in range(len(a)):
        t = a[i]
        normal = (t[1][0] - t[0][0])*(t[2][1]-t[0][1])-(t[1][1]-t[0][1])*(t[2][0]-t[0][0])

        if normal < 0:
            a[i] = t[::-1]

def area_of_intersection(a, b):
    ntriangles = a.shape[0]
    out = np.empty(ntriangles, dtype=np.float64)
    for i in range(ntriangles):
        aa = sg.Polygon(a[i])
        bb = sg.Polygon(b[i])
        out[i] = aa.intersection(bb).area
    return out

a = np.random.rand(10, 3, 2)
b = np.random.rand(10, 3, 2)
ccw(a)
ccw(b)
expected = area_of_intersection(a, b)
```

"""

import numpy as np

from numba_celltree.algorithms.sutherland_hodgman import (
    area_of_intersection,
    box_area_of_intersection,
    intersection,
    polygon_polygon_clip_area,
)
from numba_celltree.constants import FloatDType, Point, Vector

A = np.array(
    [
        [[0.98599114, 0.16203056], [0.64839124, 0.6552714], [0.44528724, 0.88567472]],
        [[0.96182162, 0.3642742], [0.03478739, 0.54268026], [0.57582971, 0.41541277]],
        [[0.32556365, 0.03800701], [0.74000686, 0.04684465], [0.89527188, 0.55061165]],
        [[0.2988294, 0.96608896], [0.01212383, 0.00144037], [0.75113002, 0.54797261]],
        [[0.06522962, 0.43735202], [0.791499, 0.5229509], [0.40651803, 0.94317979]],
        [[0.06544202, 0.16735701], [0.67916353, 0.95843272], [0.33545733, 0.86368003]],
        [[0.43129575, 0.27998206], [0.49468229, 0.75438255], [0.01542992, 0.80696797]],
        [[0.29449023, 0.32433138], [0.46157048, 0.22492393], [0.82442969, 0.75853821]],
        [[0.66113797, 0.88485505], [0.70164374, 0.24393423], [0.89565423, 0.89407158]],
        [[0.92226655, 0.82771688], [0.42243438, 0.17562404], [0.82885357, 0.17541439]],
    ],
)

B = np.array(
    [
        [[0.8141854, 0.06821897], [0.37086004, 0.49067617], [0.79810508, 0.07873283]],
        [[0.74948185, 0.8942076], [0.59654411, 0.87755533], [0.3023107, 0.68256513]],
        [[0.46670989, 0.31716127], [0.68408985, 0.75792215], [0.41437824, 0.79509823]],
        [[0.60715923, 0.67648133], [0.40045464, 0.79676831], [0.06332723, 0.69679141]],
        [[0.24057248, 0.16433727], [0.58871277, 0.05499277], [0.59144784, 0.24476056]],
        [[0.23183198, 0.41619006], [0.66566902, 0.30110111], [0.60418791, 0.60702136]],
        [[0.09393344, 0.87976118], [0.994083, 0.00532686], [0.95176396, 0.79836557]],
        [[0.89063751, 0.5880825], [0.03881315, 0.82436939], [0.61391092, 0.45027842]],
        [[0.63168954, 0.75135847], [0.8726944, 0.06387274], [0.89585471, 0.92837592]],
        [[0.94379596, 0.64164962], [0.95787609, 0.65627618], [0.6212529, 0.89153053]],
    ]
)


EXPECTED = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0262324,
        0.0,
        0.00038042,
        0.03629781,
        0.01677156,
        0.05417924,
        0.00108787,
    ]
)


def test_intersection():
    # Intersection
    a = Point(0.0, 0.0)
    V = Vector(1.0, 1.0)
    r = Point(1.0, 0.0)
    s = Point(0.0, 1.0)
    U = Vector(s.x - r.x, s.y - r.y)
    N = Vector(-U.y, U.x)
    succes, p = intersection(a, V, r, N)
    assert succes
    assert np.allclose(p, [0.5, 0.5])

    # Parallel lines, no intersection
    s = Point(2.0, 1.0)
    U = Vector(s.x - r.x, s.y - r.y)
    N = Vector(-U.y, U.x)
    succes, p = intersection(a, V, r, N)
    assert not succes


def test_clip_area():
    for a, b, expected in zip(A, B, EXPECTED):
        actual = polygon_polygon_clip_area(a, b)
        assert np.allclose(actual, expected)


def test_clip_area_no_overlap():
    a = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    b = a.copy()
    b += 2.0
    actual = polygon_polygon_clip_area(a, b)
    assert np.allclose(actual, 0)


def test_clip_area_repeated_vertex():
    a = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    # No overlap
    b = a.copy()
    b += 2.0
    actual = polygon_polygon_clip_area(a, b)
    assert np.allclose(actual, 0)

    b = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    actual = polygon_polygon_clip_area(a, b)


def test_clip_area_epsilon():
    EPS = np.finfo(FloatDType).eps
    a = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    b = np.array(
        [
            [-1.0 - EPS, -1.0 - EPS],
            [1.0 + EPS, -1.0 - EPS],
            [1.0 + EPS, 1.0 + EPS],
        ]
    )
    actual = polygon_polygon_clip_area(a, b)
    assert np.allclose(actual, 2.0)

    EPS = -EPS
    b = np.array(
        [
            [-1.0 - EPS, -1.0 - EPS],
            [1.0 + EPS, -1.0 - EPS],
            [1.0 + EPS, 1.0 + EPS],
        ]
    )
    actual = polygon_polygon_clip_area(a, b)
    assert np.allclose(actual, 2.0)


def test_area_of_intersection():
    vertices_a = A.reshape(-1, 2)
    vertices_b = B.reshape(-1, 2)
    faces_a = np.arange(len(vertices_a)).reshape(-1, 3)
    faces_b = np.arange(len(vertices_b)).reshape(-1, 3)
    indices_a = np.arange(len(faces_a))
    indices_b = np.arange(len(faces_a))
    actual = area_of_intersection(
        vertices_a, vertices_b, faces_a, faces_b, indices_a, indices_b
    )
    assert np.allclose(actual, EXPECTED)


def test_box_area_of_intersection():
    box_coords = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
        ]
    )
    vertices = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [-2.0, 0.0],
            [-2.0, 2.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 4],
        ]
    )
    indices_bbox = np.array([0, 0, 1, 1])
    indices_face = np.array([0, 1, 0, 1])
    actual = box_area_of_intersection(
        box_coords,
        vertices,
        faces,
        indices_bbox,
        indices_face,
    )
    assert np.allclose(actual, [0.5, 0.0, 0.5, 0.0])
