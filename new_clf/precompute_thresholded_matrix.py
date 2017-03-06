#!/usr/bin/python

# from threshold_embedder import *
from similarity import similarity
import numpy as np
from scipy.spatial.distance import cityblock # sad distance
from scipy.spatial.distance import sqeuclidean # ssd distance
from sklearn.metrics import pairwise_distances


class precompute_thresholded_matrix(object):
	def __init__(self, X, distance_function=sqeuclidean, threshold=30):
		# self.embedder = threshold_embedder(X, distance_function, threshold)
		self.X = X
		self.distance_function = distance_function
		self.threshold = threshold

	# def embed_ind(i):
	# 	if i % 1000 == 0:
	# 		print repr(i)
	# 	return self.embedder(X[i])
	# Xnew = [embed_ind(ii) for ii in range(len(X))]
	# Xnew = [self.embedder(X[ii]) for ii in range(len(X))]
	# return np.array(Xnew,dtype=np.float)
	def __call__(self, X):
		sim = similarity(self.threshold)
		return map(sim, pairwise_distances(X, Y=self.X,
										metric=self.distance_function,
										n_jobs=-1))
