import numpy as np
import scipy


def radon_point3(pts):
    """

    :param pts: set of d+2 or d+3 points in general position
    :return: either the unique radon point or the set of d+3 radon points of all subsets
    """
    if len(pts) == len(pts[0]) + 2:  # dirty workaround
        return radon_point_unique(pts)

    system = np.vstack((pts.transpose(), np.ones((1, len(pts)))))
    null = scipy.linalg.null_space(system)
    assert np.linalg.matrix_rank(system) == len(pts) - 2
    # here iterate over all relevant lin combinations such that
    # the sum of the lin combination factors = 1 -> another row with 1s
    # the sign of one component can be restricted, not that cool though i guess
    # the combination should have d+2 nonzero components
    # idea: for k = 3, solve an equation for each component -> lambdas, force the first lambda to be positive
    r_pts = np.empty((len(null), len(pts[0])))

    for i in range(len(null)):  # TODO: works for k =3 only!

        lambda_i = null[i][::-1]  #find a linear combination for c1*v1+c2*v2=0 -> c1=v2, c2 =v1
        assert len(lambda_i.T) == 2
        reduced_null = null.T[0] * lambda_i[0] - null.T[1] * lambda_i[1]
        #lambdas is from the the REVERSED null, therefore the indices are the way they are
        assert reduced_null[i] == 0
        null_plus = (reduced_null).clip(min=0)
        null_minus = (reduced_null).clip(max=0)
        r = system.dot(null_plus) / (null_plus.sum())  # get radon point like described by radon
        r_minus = system.dot(null_minus) / (null_minus.sum())  # get radon point like described by radon

        assert np.linalg.norm(r - r_minus) < 1e-5
        r_pts[i] = r[:-1].T
    return r_pts


def null_space(system: np.ndarray):
    """
    returns a basis of the null-space of the given system as well as the condition number to detect near deficient cases
    :param system: array of shape (n+2) x (n+1)
    :return: set of basis-vectors for the null-space as well as the condition number of the system
    """
    U, s, Vt = np.linalg.svd(system)
    non_null_columns = sum(s > 0)
    return Vt[non_null_columns:].T, max(s) / (min(s) + 1e-10)


def radon_point_unique(pts):
    """

    :type pts: array of shape (d+2,d) of points in general position
    """
    system = np.vstack((pts.transpose(), np.ones((1, len(pts))))).astype(np.double)
    # null = scipy.linalg.null_space(system)
    null, cond_number = null_space(system)
    null_plus = null.clip(min=0)
    null_minus = null.clip(max=0)
    assert len(null_plus.T) == 1
    r = system.dot(null_plus) / null_plus.sum()  # get radon point like described by radon
    r_minus = system.dot(null_minus) / null_minus.sum()
    if np.linalg.norm(r - r_minus) > 1e-10:
        print(f"{r}, {r_minus}, {r - r_minus}, {np.linalg.norm(r - r_minus)} ")
        assert False
    return r[:-1].T.astype(np.double), cond_number  # drop 1 component


if __name__ == '__main__':

    def regular_simplex(n):
        # Define the vertices of a regular n-1 dimensional simplex
        vertices = np.zeros((n, n))
        for i in range(n):
            vertices[i, i] = 1.0
        A = np.ones((n, n))
        A[1:, 1:] -= np.eye(n - 1)
        b = np.ones(n)
        b[0] = 0
        c = np.linalg.solve(A, b)
        for i in range(n):
            vertices[i, :] -= c[i]

        # Generate the set of points using a random linear combination of the vertices
        points = np.random.rand(n, n)
        points = np.vstack((points, np.ones(n)))
        points = np.linalg.solve(vertices.T, points.T).T

        return points


    d = 22
    n = d + 2
    for _ in range(10000):
        simplex = regular_simplex(d + 1)
        pt1 = np.mean(simplex, axis=0)
        pts = np.vstack((pt1, simplex))
        # print(f"mean of simplex = {pt1}\n radon point = {radon_point_unique(pts)}\n diff = {np.linalg.norm(radon_point_unique(pts) - pt1)}")
        assert np.linalg.norm(radon_point_unique(pts) - pt1) < 1e-12

    d = 22  # dims
    n = d + 3  # amount of hypotheses
    k = 0  # outlier hypotheses
    h = 1  # height
    np.random.seed(42)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print()
    pts = np.vstack((1000 * np.random.randn(n - k, d), 0.01 * np.random.randn(k, d) + 50))

    faston_points = radon_point3(pts)
    faston_points2 = radon_point3(faston_points)
    print(
        f"mean of radon   points: {faston_points.mean(axis=0)}\nmean of radon^2 points: {faston_points2.mean(axis=0)}")
    orig_mean = pts.mean(axis=0)
