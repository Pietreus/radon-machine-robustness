import numpy as np

from radon_machine.radon_point.fast_radon_points import radon_point_unique, radon_point3


def iterated_radon_point(points, radon_num, height, sigma=1e-5, shuffle = True):
    if shuffle:
        np.random.shuffle(points)
    condition_numbers = []
    assert len(points) == radon_num ** height
    for _ in range(height):
        points += np.random.randn(*points.shape) * (sigma * points.std(axis=0))
        points, cond_nums = radon_aggregate(points, radon_num)
        condition_numbers.append(cond_nums)
    return points[0], [item for sublist in condition_numbers for item in sublist]


def radon_aggregate(pts, r):
    radons = []
    condition_numbers = []
    for i in range(0, len(pts), r):
        rad, cond_number = radon_point_unique(pts[i:(i + r)])
        radons.append(rad[0])  # it's just a unique radon point
        condition_numbers.append(cond_number)
    return np.array(radons), condition_numbers


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
