import scipy
import numpy as np
import cdd as pcdd
from scipy.spatial import ConvexHull


def radon_point(pts):
    return hull_to_points(radon_hull(pts))


def radon_hull(pts):
    hull = v_representation(ConvexHull(pts).points).get_inequalities()
    for i in range(len(pts)):
        hull = np.vstack((hull, v_representation(ConvexHull(np.delete(pts, i, axis=0)).points).get_inequalities()))
    return hull


# TODO
# https://math.stackexchange.com/questions/4425296/intersection-of-convex-hulls
def hull_to_points(hull):
    mat = pcdd.Matrix(hull, number_type="float")
    mat.rep_type = pcdd.RepType.INEQUALITY
    poly_intersection = pcdd.Polyhedron(mat)

    # get the vertices; they are given in a matrix prepended by a column of ones
    v_intersection = poly_intersection.get_generators()
    # get rid of the column of ones
    return np.array([
        v_intersection[i][1:] for i in range(len(v_intersection))
    ])


def v_representation(pts):
    v = np.column_stack((np.ones(len(pts)), pts))
    mat = pcdd.Matrix(v, number_type="fraction")  # use fractions if possible
    mat.rep_type = pcdd.RepType.GENERATOR
    return pcdd.Polyhedron(mat)


if __name__ == '__main__':
    d = 7  # dims
    n = d + 4  # amount of hypotheses
    k = 0  # outlier hypotheses
    h = 1  # height
    np.random.seed(11905150)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    pts = np.vstack((10 * np.random.randn(n - k, d), 0.01 * np.random.randn(k, d) + 50))
    # pts = np.array([[2,0],[0,1],[0,3],[2,4],[3,3]])
    # print(np.hstack((pts, np.zeros((d+2, 1)),np.zeros((d+2, 1)))))
    # print(np.linalg.lstsq(np.hstack((pts, np.zeros((d+2, 1)), np.zeros((d+2, 1)))), np.array([0,0,0,0,0])))
    system = np.vstack((pts.transpose(), np.ones((1, n))))
    np.linalg.lstsq(system, np.zeros((d + 1, 1)))
    # linear combinations of null_space solutions??
    # linear combinations of 2-3.....
    l, v, r = np.linalg.svd(system)
    np.dot(system, r)
    null = scipy.linalg.null_space(system)
    print(null)
    # all radon partitions:

    # idea: find all nullstellen for each variable in the linear combinations, then the "surrounding area" can be used to identify all radon partitions
    vec = np.empty((1, n))
    for i in np.linspace(0, 1, num=1000):
        print("\n\npartition:")
        vec = np.vstack((vec, np.sign(null.T[0] * i + null.T[1] * (1 - i))))
    print(np.unique(vec, axis=0))

    # all solutions to the radon point problem are in the nullspace of the matrix
    # the 'defekt'/nullity of a matrix then show how many degrees of freedom the radon point actually has
    # this is because the null_space function gives base vectors for the nullspace, any linear combination of these vectors
    # then gives all 'directions' in the nullspace.
    # Using the directions of the nullspace, radon partitions can be constructed as introduced by radon.
    # Interesting observations: each time the sign of a coordinate changes, the given point in the original set swaps sets in the partition.
    # what is the pipeline?:
    # take a system of linear equations with n+1 obs.
    # find its nullspace
    # all linear combinations of this nullspace constitute partitions that produce radon points.
    # how can I make sure I "find" all linear combinations to get all possible partitions?
    # just combinations with length 1 would be a unit circle, of which i don't want one half because its just the same as the other half with swapped signs.
    # this would allow me 0 degs of freedom for n=d+2, 1 for n=d+3, and so on and so forth.

    # shouldnt the signs or their combinations be immediately clear

# sols = np.linalg.svd(system)#, np.zeros((d+1, 1)))
# print(sols)
# print(ConvexHull(pts).points)
# print(radon_hull(pts))
# hull = hull_to_points(radon_hull(pts))
# print("Radon hull:", end=" ")
# print(hull)
# mesh = go.Mesh3d(x=hull.T[0], y=hull.T[1], z=hull.T[2], color='red', opacity=0.5, showscale=False)
#
# points = go.Scatter3d(x=pts.T[0], y=pts.T[1], z=pts.T[2], line=dict(color='black', width=0))
# radon_point = go.Scatter3d(x=hull.T[0], y=hull.T[1], z=hull.T[2], line=dict(color='blue', width=0))
# fig = go.FigureWidget(data=[mesh, points, radon_point])
# fig.show()
# print(ConvexHull(sect).points[0])
# these are the vertices of the intersection; it remains to take
# the convex hull
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
