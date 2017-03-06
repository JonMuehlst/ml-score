import pyximport
pyximport.install()
from fastdist import pairwise
from fastcdist import cpairwise
import numpy as np
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, dice, euclidean, hamming, jaccard, kulsinski, mahalanobis, matching, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule

A0 = np.random.randn(1, 200)
A = np.random.randn(100, 200)


D1 = np.sqrt(np.square(A[np.newaxis,:,:]-A[:,np.newaxis,:]).sum(2))
dist = pairwise(euclidean)
cdist = cpairwise(euclidean)
D2 = dist(A)
D3 = cdist(A0, A)

print repr(len(D3[0]))
print np.allclose(D1, D2)
print repr(len(D2)), repr(len(D2[0]))
# True
