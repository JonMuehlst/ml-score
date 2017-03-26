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
        kernel=self.myKernel(gamma=self.gamma, distance=self.distance),
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
        self.clf = BaggingClassifier(mysvm, max_samples=1.0 / n_estimators, n_estimators=10, n_jobs=-1)
        return self.clf.fit(X,y)

    """ """
    class myKernel(Kernel):
        """
        Radial Basis Function kernel, defined as unnormalized Gaussian PDF

            K(x, y) = e^(-g||x - y||^2)

        where:
            g = gamma
        """

        def __init__(self, gamma=None, distance='sqeuclidean'):
            self._gamma = gamma
            self._distance = distance

        def _compute(self, data_1, data_2):
            if self._gamma is None:
                # libSVM heuristics
                self._gamma = 1./data_1.shape[1]

            dists_mat = pairwise_distances(X=data_1, Y=data_2, metric=self._distance, n_jobs=-1)
            return np.exp(-self._gamma * dists_mat)

        def dim(self):
            return np.inf


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


"""
Base classes and methods used by all kernels
"""

#https://raw.githubusercontent.com/gmum/pykernels/master/pykernels/base.py

#__author__ = 'lejlot'

import numpy as np
from abc import abstractmethod, ABCMeta

class Kernel(object):
    """
    Base, abstract kernel class
    """
    __metaclass__ = ABCMeta

    def __call__(self, data_1, data_2):
        return self._compute(data_1, data_2)

    @abstractmethod
    def _compute(self, data_1, data_2):
        """
        Main method which given two lists data_1 and data_2, with
        N and M elements respectively should return a kernel matrix
        of size N x M where K_{ij} = K(data_1_i, data_2_j)
        """
        raise NotImplementedError('This is an abstract class')

    def gram(self, data):
        """
        Returns a Gramian, kernel matrix of matrix and itself
        """
        return self._compute(data, data)

    @abstractmethod
    def dim(self):
        """
        Returns dimension of the feature space
        """
        raise NotImplementedError('This is an abstract class')

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def __add__(self, kernel):
        return KernelSum(self, kernel)

    def __mul__(self, value):
        if isinstance(value, Kernel):
            return KernelProduct(self, value)
        else:
            if isinstance(self, ScaledKernel):
                return ScaledKernel(self._kernel, self._scale * value)
            else:
                return ScaledKernel(self, value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __div__(self, scale):
        return ScaledKernel(self, 1./scale)

    def __pow__(self, value):
        return KernelPower(self, value)

class KernelSum(Kernel):
    """
    Represents sum of a pair of kernels
    """

    def __init__(self, kernel_1, kernel_2):
        self._kernel_1 = kernel_1
        self._kernel_2 = kernel_2

    def _compute(self, data_1, data_2):
        return self._kernel_1._compute(data_1, data_2) + \
               self._kernel_2._compute(data_1, data_2)

    def dim(self):
        # It is too complex to analyze combined dimensionality, so we give a lower bound
        return max(self._kernel_1.dim(), self._kernel_2.dim())

    def __str__(self):
        return '(' + str(self._kernel_1) + ' + ' + str(self._kernel_2) + ')'


class KernelProduct(Kernel):
    """
    Represents product of a pair of kernels
    """

    def __init__(self, kernel_1, kernel_2):
        self._kernel_1 = kernel_1
        self._kernel_2 = kernel_2

    def _compute(self, data_1, data_2):
        return self._kernel_1._compute(data_1, data_2) * \
               self._kernel_2._compute(data_1, data_2)

    def dim(self):
        # It is too complex to analyze combined dimensionality, so we give a lower bound
        return max(self._kernel_1.dim(), self._kernel_2.dim())

    def __str__(self):
        return '(' + str(self._kernel_1) + ' * ' + str(self._kernel_2) + ')'


class KernelPower(Kernel):
    """
    Represents natural power of a kernel
    """

    def __init__(self, kernel, d):
        self._kernel = kernel
        self._d = d
        if not isinstance(d, int) or d<0:
            raise Exception('Kernel power is only defined for non-negative integer degrees')

    def _compute(self, data_1, data_2):
        return self._kernel._compute(data_1, data_2) ** self._d

    def dim(self):
        # It is too complex to analyze combined dimensionality, so we give a lower bound
        return self._kernel.dim()

    def __str__(self):
        return str(self._kernel) + '^' + str(self._d)


class ScaledKernel(Kernel):
    """
    Represents kernel scaled by a float
    """

    def __init__(self, kernel, scale):
        self._kernel = kernel
        self._scale = scale
        if scale < 0:
            raise Exception('Negation of the kernel is not a kernel!')

    def _compute(self, data_1, data_2):
        return self._scale * self._kernel._compute(data_1, data_2)

    def dim(self):
        return self._kernel.dim()

    def __str__(self):
        if self._scale == 1.0:
            return str(self._kernel)
        else:
            return str(self._scale) + ' ' + str(self._kernel)


class GraphKernel(Kernel):
    """
    Base, abstract GraphKernel kernel class
    """
    pass
