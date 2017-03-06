#!/usr/bin/python

import numpy as np
# from distance import *
from scipy.spatial.distance import cityblock # sad distance
from scipy.spatial.distance import sqeuclidean # ssd distance


class similarity(object):
	def __init__(self, distance_function=sqeuclidean, threshold=30):
		# self.distance_function=distance_function
		self.threshold=threshold

	# def __call__(self, v1, v2):
	# 	distance = self.distance_function(v1,v2)
	# 	#print 'Distance: ' + repr(distance)
	# 	return np.subtract(1,(np.minimum(self.threshold, distance)/np.float(self.threshold)))

	def __call__(self, v):
		return map(self.sim, v)

	def sim(self, distance):
		return np.subtract(1,(np.minimum(self.threshold, distance)/np.float(self.threshold)))
