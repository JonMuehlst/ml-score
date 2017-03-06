#!/usr/bin/python

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_random_state, shuffle
from sklearn.metrics import accuracy_score
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


class emb_svm(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold_ind=0, distance='sqeuclidean', penalty='l2',
                    loss='squared_hinge', dual=True, tol=0.0001,
                    C=1.0, multi_class='ovr', fit_intercept=True,
                    intercept_scaling=1, class_weight=None, verbose=0,
                    random_state=None, max_iter=1000):
        self.threshold_ind = threshold_ind
        self.distance_type = distance
        self.distance_function = get_distance_funcs()[distance]
        self.random_state_ = random_state
        self.svm = svm.LinearSVC(loss='hinge', C=C)


    def fit(self, X, y):
        # t00 = time()
        self.threshold = generate_threshold_values(X)[self.threshold_ind]
        self.precomputer = precompute_thresholded_matrix(X,
                                    distance_function=self.distance_function,
                                    threshold=self.threshold)

        self.random_state_ = check_random_state(self.random_state_)

        # print "fit time:", round(time()-t00, 10), "s"

        self.svm = self.svm.fit(self.precomputer(X),y)

        return self

    """ """
    def score(self,X,y):

        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)


    """ """
    def predict(self, X):
        return self.svm.predict(self.precomputer(X))

    def get_params(self, deep=True):
        return dict(self.svm.get_params(deep), **{'distance':self.distance_type, 'threshold_ind':self.threshold_ind})

    def set_params(self, **parameters):
        t_param_keys = { 'threshold_ind', 'distance' }
        t_params = { key:value for key,value in parameters.items() if key in t_param_keys }
        svm_params = { key:value for key,value in parameters.items() if key not in t_param_keys }
        for parameter, value in t_params.items():
            self.setattr(parameter, value)
        self.svm.set_params(svm_params)
        return self
