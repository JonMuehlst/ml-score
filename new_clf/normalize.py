#!/usr/bin/env python
#!/usr/bin/python

import numpy as np

def min_max_feature_values(X):

	return map(min, zip(*X)), map(max, zip(*X))


def normalize_sample(sample, (min_values, max_values)):

	ans = np.zeros(len(sample))
	for ii in range(len(sample)):
	    ans[ii] = normalization_formula(sample[ii],min_values[ii],max_values[ii])

	return ans

def normalization_formula(value, min_value, max_value):
	if max_value == 0:
		return 0
	return np.float(np.float((value - min_value)) / (np.float(max_value - min_value)))

def normalize_matrix(X):

    # M = samples
    # N = features
    M = len(X)
    N = len(X[0])

    Xnew = np.zeros((M,N))
    min_values, max_values = min_max_feature_values(X)

    for ii in range(M):
    	Xnew[ii] = normalize_sample(X[ii], (min_values, max_values))

    return Xnew

def normalize_matrix_by_min_max(X, (min_values, max_values)):
	Xnorm = np.zeros((len(X),len(X[0])))

	for ii in range(len(X)):
		Xnorm[ii] = normalize_sample(X[ii], (min_values, max_values))
	return Xnorm
