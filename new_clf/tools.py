""" scipy.spatial.distance.pdist(X, metric='euclidean',
                                 p=2, w=None, V=None, VI=None)

See distance function at the bottom
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, dice, euclidean, hamming, jaccard, kulsinski, mahalanobis, matching, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule

def generate_threshold_values(X, metric='euclidean',
                                 p=2, w=None, V=None, VI=None):

    distances = np.sort(pdist(X,metric=metric, p=p, w=w, V=V, VI=VI))
    # print repr(distances)
    size = int((len(X)*(len(X)-1))/2)
    jumps = int(size/11)
    t_vals = np.array([distances[t] for t in range(jumps-1,size-jumps,jumps)])
    # return np.unique(t_vals)
    return t_vals

def get_distance_funcs():
    return {'braycurtis':braycurtis,
    'canberra':canberra,
    'chebyshev':chebyshev,
    'cityblock':cityblock,
    'correlation':correlation,
    'cosine':cosine,
    'dice':dice,
    'euclidean':euclidean,
    'hamming':hamming,
    'jaccard':jaccard,
    'kulsinski':kulsinski,
    'mahalanobis':mahalanobis,
    'matching':matching,
    'minkowski':minkowski,
    'rogerstanimoto':rogerstanimoto,
    'russellrao':russellrao,
    'seuclidean':seuclidean,
    'sokalmichener':sokalmichener,
    'sokalsneath':sokalsneath,
    'sqeuclidean':sqeuclidean,
    'yule':yule}
