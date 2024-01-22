from copy import deepcopy

import cython
import matplotlib.pyplot as plt
import numpy as np

cimport numpy as np

np.import_array()
ctypedef np.float_t DTYPE_t
DTYPE = np.float64

def func_Kmeanspp(np.ndarray X, int ncls):
    '''
    intialized the centroids for K-means++
    '''
    cdef double [:] dist_min 
    cdef int ndata = X.shape[0]
    cdef Py_ssize_t nfeature = X.shape[1]
    cdef Py_ssize_t icls, icls_pre, imin
    cdef int imax 
    centroids = np.zeros((ncls, nfeature), dtype=DTYPE)
    centroids[0, :] = X[np.random.randint(ndata), :]
    for icls in range(1, ncls):
        distances = np.zeros((icls, ndata))
        # print(distances.shape)
        dist_min = np.zeros(ndata)
        for icls_pre in range(icls):
            distances[icls_pre, :] = np.linalg.norm(X - centroids[icls_pre, :], axis=1)
        idx_min = np.argmin(distances, axis=0)
        for imin in range(len(idx_min)):
            dist_min[imin] = distances[idx_min[imin], imin]
        imax = np.argmax(dist_min) % ndata
        centroids[icls, :] = X[imax, :]
    return centroids

## initialization algorithm
def func_Kmeans(np.ndarray X, int ncls, str init='k-means++', int max_iter=300):
    cdef int error
    cdef int cluster 
    cdef int ndata = X.shape[0]
    cdef int nfeature = X.shape[1]
    cdef Py_ssize_t i, icls_1, icls_2 
    if init == 'random':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        centers_init = np.random.randn(ncls, nfeature)*std + mean
    else:
        centers_init = func_Kmeanspp(X, ncls)
    centers = deepcopy(centers_init)
    clusters = np.zeros(ndata)
    distances = np.zeros((ndata, ncls))
    for i in range(max_iter):
        for icls_1 in range(ncls):
            distances[:, icls_1] = np.linalg.norm(X - centers[icls_1], axis=1)
        clusters = np.argmin(distances, axis=1)
        centers_pre = deepcopy(centers)
        for icls_2 in range(ncls):
            if (clusters == icls_2).any():
                centers[icls_2, :] = np.mean(X[clusters == icls_2], axis=0)
        error = np.linalg.norm(centers - centers_pre)
        if error == 0:
            break
    return clusters, centers,distances