import numpy as np
from scipy.spatial import cKDTree
from numba import njit


@njit
def assign_indexes(x_shape, indexes, dists):
    dists_tensor = np.zeros(x_shape)
    for i in range(len(indexes)):
        dists_tensor[indexes[i][0], indexes[i][1], indexes[i][2]] = dists[i]
    return dists_tensor


@njit
def ndindex_2d(shape):
    n1, n2 = shape
    arr = np.empty((n1 * n2, 2), dtype=np.int64)
    for i in range(n1):
        for j in range(n2):
            arr[i*n2+j] =  (i, j)
    return arr


@njit
def ndindex_3d(shape):
    n1, n2, n3 = shape
    arr = np.empty((n1 * n2 * n3, 3), dtype=np.int64)
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                arr[i*n2*n3+j*n3+k] =  (i, j, k)
    return arr


# Implementation by @stefanv (Stefan van der Walt)
# https://github.com/numpy/numpy/issues/1234
# Warning: does not output results in C order
# the specialised numba ndindex functions are faster
def fast_ndindex(shape):
    return np.indices(shape).T.reshape(-1, len(shape))


# more generic but slower
def distance_grid(points, x_shape):
    if points is None or len(points) == 0:
        return np.ones(x_shape)
    points = np.asarray(points)
    x_shape = np.asarray(x_shape)
    
    tree = cKDTree(points, balanced_tree=False)
    indexes = fast_ndindex(x_shape)
    if ub is None:
        ub = int(np.square(np.max(x_shape) * 0.05))
    dists = tree.query(indexes, n_jobs=-1, distance_upper_bound=ub)[0]
    dists = np.minimum(dists, ub) / ub

    return assign_indexes(x_shape, indexes, dists)


# actually used version
def distance_grid_3d(points, x_shape, indexes=None, ub=None):
    if points is None or len(points) == 0:
        return np.ones(x_shape)
    points = np.asarray(points)
    x_shape = np.asarray(x_shape)
    
    tree = cKDTree(points, balanced_tree=False)
    # option: provide indexes in arguments if they are always the same
    if indexes is None:
        indexes = ndindex_3d(x_shape)
    if ub is None:
        ub = int(np.square(np.max(x_shape) * 0.05))
    dists = tree.query(indexes, n_jobs=-1, distance_upper_bound=ub)[0]
    dists = np.minimum(dists, ub) / ub

    # Indexes are in C order, so dists too
    # This is why we can directly reshape
    return np.reshape(dists, x_shape)


# Experimental version
# Faster than distance_grid_3d for low point counts (~<10000)
# distance_grid_3d scales better with more points
# Kept only for reference
@njit
def distance_grid_3d_fast(points, x_shape):
    ub_sqrt = np.ceil(max(x_shape) * 0.05)
    ub = int(np.square(max(x_shape) * 0.05))

    arr = np.ones(x_shape)
    for pidx in range(points.shape[0]):
        p = points[pidx]
        arr[p[0], p[1], p[2]] = 0
        for i in range(-ub, ub + 1):
            x = p[0] + i
            if x >= 0 and x < x_shape[0]:
                i_sqr = i * i
                abs_i = abs(i)
                for j in range(-ub + abs_i, ub - abs_i + 1):
                    y = p[1] + j
                    if y >= 0 and y < x_shape[1]:
                        j_sqr = j * j
                        abs_j = abs(j)
                        for k in range(-ub + abs_i + abs_j, ub - abs_i - abs_j + 1):
                            z = p[2] + k
                            if z >= 0 and z < x_shape[2]:
                                k_sqr = k * k
                                arr[x, y, z] = min(min(np.sqrt(i_sqr + j_sqr + k_sqr), ub) / ub, arr[x, y, z])
    return arr

# 2d version of distance_grid_3d for reference
def distance_grid_2d(points, x_shape, indexes=None, ub=None):
    if points is None or len(points) == 0:
        return np.ones(x_shape)
    points = np.asarray(points)
    x_shape = np.asarray(x_shape)
    
    tree = cKDTree(points, balanced_tree=False)
    # option: provide indexes in arguments if they are always the same
    if indexes is None:
        indexes = ndindex_2d(x_shape)
    if ub is None:
        ub = int(np.square(np.max(x_shape) * 0.05))
    dists = tree.query(indexes, n_jobs=-1, distance_upper_bound=ub)[0]
    dists = np.minimum(dists, ub) / ub

    # Indexes are in C order, so dists too
    # This is why we can directly reshape
    return np.reshape(dists, x_shape)