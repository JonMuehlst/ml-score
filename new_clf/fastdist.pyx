import numpy as np
cimport numpy as np
cimport cython
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, dice, euclidean, hamming, jaccard, kulsinski, mahalanobis, matching, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule

# don't use np.sqrt - the sqrt function from the C standard library is much
# faster
from libc.math cimport sqrt

# disable checks that ensure that array indices don't go out of bounds. this is
# faster, but you'll get a segfault if you mess up your indexing.
@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)

cdef class pairwise(object):

#    def       d(self): return 0
#    cdef  int c(self):

  cdef object distance

  def __init__(self, distance):
    self.distance = distance

  def __call__(self, double [:, :] A):

      # declare C types for as many of our variables as possible. note that we
      # don't necessarily need to assign a value to them at declaration time.
      cdef:
          # Py_ssize_t is just a special platform-specific type for indices
          Py_ssize_t nrow = A.shape[0]
          Py_ssize_t ncol = A.shape[1]
          Py_ssize_t ii, jj, kk

          # this line is particularly expensive, since creating a numpy array
          # involves unavoidable Python API overhead
          np.ndarray[np.float64_t, ndim=2] D = np.zeros((nrow, nrow), np.double)

          double tmpss, diff

      # another advantage of using Cython rather than broadcasting is that we can
      # exploit the symmetry of D by only looping over its upper triangle
      for ii in range(nrow):
          for jj in range(ii + 1, nrow):
              # we use tmpss to accumulate the SSD over each pair of rows
              #tmpss = 0
              #for kk in range(ncol):
              #    diff = A[ii, kk] - A[jj, kk]
              #    tmpss += diff * diff
              #tmpss = sqrt(tmpss)
              # D[ii, jj] = tmpss
              # D[jj, ii] = tmpss  # because D is symmetric
              dist = self.distance(A[ii], A[jj])
              D[ii, jj] = dist
              D[jj, ii] = dist

      return D
