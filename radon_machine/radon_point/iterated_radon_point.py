import numpy as np

from radon_machine.radon_point.fast_radon_points import radon_point_unique, radon_point3


def iterated_radon_point(points, radon_num, height, sigma=1e-5):
    np.random.shuffle(points)
    assert len(points) == radon_num ** height
    for _ in range(height):
        points += np.random.randn(*points.shape) * (sigma * points.std(axis=0))
        points = radon_aggregate(points, radon_num)
    return points[0]


def radon_aggregate(pts, r):
    radons = []
    for i in range(0, len(pts), r):
        rad = radon_point_unique(pts[i:(i + r)])
        radons.append(rad[0])  # its just a unique radon point
    return np.array(radons)


############################# for d+3 radon machine #############################

def iterated_radon3_point(points, radon_num_plus_one, height, sigma=1e-5):
    assert len(points) == radon_num_plus_one ** height
    for _ in range(height):
        points += np.random.randn(*points.shape) * sigma * points.std(axis=0)
        points = radon3_aggregate(points, radon_num_plus_one)
    return points[0]


def radon3_aggregate(pts, r):
    radons = []
    for i in range(0, len(pts), r):
        rad = radon_point3(pts[i:(i + r)])
        radons.append(radon_selection(pts[i:(i + r)], rad))
    return np.array(radons)


def radon_selection(pts, rad):
    med = np.median(pts, axis=0)
    mini = -1
    mindist = np.inf
    for i in range(len(rad)):
        dist = np.linalg.norm(rad[i] - med)
        if mindist > dist:
            mindist = dist
            mini = i
    return rad[mini]


if __name__ == '__main__':
    d = 8
    h = 4
    n = (d + 3) ** h
    pts = np.random.randn(n, d)
    print(iterated_radon3_point(pts, d + 3, h))
