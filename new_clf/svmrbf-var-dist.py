#!/usr/bin/python

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_random_state, shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.ensemble import BaggingClassifier
import sys
from threshold_embedder import threshold_embedder
from precompute_thresholded_matrix import precompute_thresholded_matrix
from scipy.spatial.distance import cityblock # sad distance
from scipy.spatial.distance import sqeuclidean # ssd distance

from time import time, ctime
from tools import get_distance_funcs, generate_threshold_values
# from zmemb.distance import ssd_distance, sad_distance
# from WeightVector import WeightVector
# from normalize import *
# from distance import *
# import itertools
# from multiprocessing import Pool

from sklearn import svm

"""
class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
shrinking=True, probability=False, tol=0.001,
cache_size=200, class_weight=None, verbose=False,
max_iter=-1, decision_function_shape=None, random_state=None)
"""
class svmrbf-var-dist(BaseEstimator,ClassifierMixin):

    """
    Valid values for metric (distance) are:
    From scikit-learn:
    [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’].
    These metrics support sparse matrix inputs.

    From scipy.spatial.distance:
    [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’,
    ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’,
    ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
    ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    See the documentation for scipy.spatial.distance
    for details on these metrics.
    These metrics do not support sparse matrix inputs.
    """
    def __init__(self, #threshold_ind=0,
                    distance='sqeuclidean',
                    C=1.0, degree=3, gamma='auto', coef0=0.0,
                    shrinking=True, probability=False, tol=0.001,
                    cache_size=200, class_weight=None, verbose=False,
                    max_iter=-1, decision_function_shape=None, random_state=None
                    ):
        # self.threshold_ind = threshold_ind
        self.distance=distance
        self.C=C
        self.degree=degree
        self.gamma=gamma
        self.coef0=coef0
        self.shrinking=shrinking
        self.probability=probability
        self.tol=tol
        self.cache_size=cache_size
        self.class_weight=class_weight
        self.verbose=verbose
        self.max_iter=max_iter
        self.decision_function_shape=decision_function_shape
        self.random_state=random_state
        self.clf = svm.SVC()



    def fit(self, X, y):
        # t00 = time()
        # self.threshold = generate_threshold_values(X)[self.threshold_ind]
        # self.precomputer = precompute_thresholded_matrix(X,
                                    # distance_function=self.distance_function
                                    #, threshold=self.threshold
                                    # )

        self.random_state_ = check_random_state(self.random_state_)

        # print "fit time:", round(time()-t00, 10), "s"

        mysvm = svm.SVC(
        kernel=self.kernel,
        C=self.C,
        degree=self.degree,
        gamma=self.gamma,
        coef0=self.coef0,
        shrinking=self.shrinking,
        probability=self.probability,
        tol=self.tol,
        cache_size=self.cache_size,
        class_weight=self.class_weight,
        verbose=self.verbose,
        max_iter=self.max_iter,
        decision_function_shape=self.decision_function_shape,
        random_state=self.random_state
        )
        self.clf = BaggingClassifier(mysvm, max_samples=1.0 / n_estimators, n_estimators=10)
        return self.clf.fit(X,y)

    """ """
    def kernel(self, A, B):

        if self.gamma is None:
            # libSVM heuristics
            self.gamma = 1./A.shape[1]

        # distance_function = get_distance_funcs()[self.distance]
        dists_mat = pairwise_distances(X=A, Y=B, metric=self.distance, n_jobs=-1)
        return np.exp(-self.gamma * dists_mat)


    """ """
    def score(self,X,y):

        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)


    """ """
    def predict(self, X):
        return self.clf.predict(X)

    def get_params(self, deep=True):
        return {
        'distance':self.distance,
        'C':self.C,
        'degree':self.degree,
        'gamma':self.gamma,
        'coef0':self.coef0,
        'shrinking':self.shrinking,
        'probability':self.probability,
        'tol':self.tol,
        'cache_size':self.cache_size,
        'class_weight':self.class_weight,
        'verbose':self.verbose,
        'max_iter':self.max_iter,
        'decision_function_shape':self.decision_function_shape,
        'random_state':self.random_state,
        # , 'threshold_ind':self.threshold_ind
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self
