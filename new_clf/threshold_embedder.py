#!/usr/bin/python

## from distance import *
from similarity import similarity
from normalize import *
# from distance import *
from scipy.spatial.distance import cityblock # sad distance
from scipy.spatial.distance import sqeuclidean # ssd distance
from scipy.spatial.distance import cdist

import numpy as np
# import itertools
from sklearn.metrics import pairwise_distances

class threshold_embedder(object):
	def __init__(self, X_input, distance_function = sqeuclidean, threshold = 30):
		#self.X = normalize_matrix(X_input)
		self.X = X_input
		self.distance_function = distance_function
		self.threshold = threshold
		# self.min_values, self.max_values = min_max_feature_values(X_input)


	def __call__(self, sample):
		# kwargs={'distance_function': self.distance_function, 'threshold':self.threshold}
		sim = similarity(self.distance_function, self.threshold)
		# return cdist([sample], self.X, metric=sim)[0]
		return pairwise_distances(np.array([sample]), Y=self.X, metric=sim, n_jobs=-1)[0]
