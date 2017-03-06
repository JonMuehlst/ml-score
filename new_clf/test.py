#!/usr/bin/env python
#!/usr/bin/python

# from distance import *
from similarity import *
from normalize import min_max_feature_values, normalize_sample, normalize_matrix, normalize_matrix_by_min_max
# from threshold_embedder import *
from precompute_thresholded_matrix import *
from tools import generate_threshold_values
import unittest
from sklearn import preprocessing
import numpy as np
from scipy.spatial.distance import cityblock # sad distance
from scipy.spatial.distance import sqeuclidean # ssd distance


### Test the distance functions
class DistanceTest(unittest.TestCase):

    def test_ssd_distance(self):
        v1 = [1, 2, 3, 4, 5]
        v2 = [11, 12, 13, 14, 15]

        distance = sqeuclidean(v1,v2)
        self.assertEqual(distance, 500)

    def test_sad_distance(self):

        v1 = [1, 2, 3, 4, 5]
        v2 = [11, 12, 13, 14, 15]

        distance = cityblock(v1,v2)

        self.assertEqual(distance, 50)


### Test the similarity function
# class SimilarityTest(unittest.TestCase):
#
#     def test_PL_similarity_1(self):
#         v1 = [10, 13, 16, 15, 18]
#         v2 = [11, 12, 13, 14, 15]
#
#         distance_val = sqeuclidean(v1,v2)
#
#         #print 'distance val: ' + repr(distance_val)
#         sim = similarity(sqeuclidean, 9.1916393868289798519)
#         sim_val = sim(v1, v2)
#         expected = 1-min(distance_val,9.1916393868289798519)/9.1916393868289798519
#
#         #print 'similarity val: ' + repr(sim_val)
#
#         self.assertEqual(sim_val,expected)
#
#     def test_PL_similarity_2(self):
#         v1 = [11, 12, 13, 14, 15]
#         v2 = [11, 12, 13, 14, 15]
#
#         distance_val = sqeuclidean(v1,v2)
#
#         #print 'distance val: ' + repr(distance_val)
#         sim = similarity(sqeuclidean, 9.1916393868289798519)
#         sim_val = sim(v1, v2)
#         expected = 1-min(distance_val,9.1916393868289798519)/9.1916393868289798519
#
#         #print 'similarity val: ' + repr(sim_val)
#
#         self.assertEqual(sim_val,expected)

### Test the normalizer
class NormalizeTest(unittest.TestCase):

    def test_min_max_feature_values(self):
        X = [[10, 13, 16, 15, 18], [11, 12, 13, 14, 15]]

        exptected_min = [10, 12, 13, 14, 15]
        exptected_max = [11, 13, 16, 15, 18]

        min_values, max_values = min_max_feature_values(X)
        #print repr(min_max_feature_values(X))

        for ii in range(len(X[0])):
            self.assertTrue((exptected_min[ii] == min_values[ii]) and (exptected_max[ii] == max_values[ii]))

    def test_normalize_sample(self):
        X = [[10, 13, 16, 15, 18], [11, 12, 13, 14, 15]]

        min_values, max_values = min_max_feature_values(X)
        normalized_sample = normalize_sample(X[0], (min_values, max_values))

        expected_sample = [0, 1, 1, 1, 1]

        #print 'original_sample: ' + repr(X[0])
        #print 'exptected_sample: ' + repr(expected_sample)

        for ii in range(len(expected_sample)):
        	self.assertEqual(normalized_sample[ii],expected_sample[ii])

    def test_normalize_matrix(self):
        X = [[10, 13, 16, 15, 18], [11, 12, 13, 14, 15]]

        normalized_matrix = normalize_matrix(X)
        expected_matrix = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0]]

        # print 'original_matrix: ' + repr(X)
        # print 'normalized_matrix: ' + repr(normalized_matrix)

        for ii in range(len(X)):
            for jj in range(len(X[0])):
            	self.assertEqual(normalized_matrix[ii][jj], expected_matrix[ii][jj])

    def test_normalize_matrix_by_min_max(self):
        X = [[10, 13, 16, 15, 18], [11, 12, 13, 14, 15]]
        min_values, max_values = min_max_feature_values(X)

        normalized_matrix = normalize_matrix_by_min_max(X, (min_values, max_values))
        expected_matrix = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0]]

        # print 'original_matrix: ' + repr(X)
        # print 'normalized_matrix: ' + repr(normalized_matrix)

        for ii in range(len(X)):
            for jj in range(len(X[0])):
            	self.assertEqual(normalized_matrix[ii][jj], expected_matrix[ii][jj])

    def test_sklearn_minmax_scaler(self):
        min_max_scaler = preprocessing.MinMaxScaler()

        X = np.array([[10., 13., 16., 15., 18.], [11., 12., 13., 14., 15.]])

        min_max_scaler.fit(X)

        normalized_matrix = min_max_scaler.transform(X)
        expected_matrix = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0]]

        # print 'original_matrix: ' + repr(X)
        # print 'normalized_matrix: ' + repr(normalized_matrix)

        for ii in range(len(X)):
            for jj in range(len(X[0])):
            	self.assertEqual(normalized_matrix[ii][jj], expected_matrix[ii][jj])


### Test the embedder
# class EmbedderTest(unittest.TestCase):
#
#     def test_embedder(self):
#         X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
#         sample = [11, 12, 13, 14, 15]
#         min_values, max_values = min_max_feature_values(X)
#         X = normalize_matrix(X)
#         sample = normalize_sample(sample, (min_values, max_values))
#         embedder = threshold_embedder(X, sqeuclidean, 10)
#
#         actual = embedder(sample)
#         expected = [0.875, 0.96875, 1., 0.96875, 0.875]
#         for ii in range(5):
#             self.assertEqual(actual[ii], expected[ii])


### Test the pre compute function
class PrecomputeTest(unittest.TestCase):

    def test_precompute(self):

        X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
        X = normalize_matrix(X)
        actual = precompute_thresholded_matrix(X, sqeuclidean, 30)(X)

        expected = [[ 1.        ,  0.98958333,  0.95833333,  0.90625   ,  0.83333333],
                    [ 0.98958333,  1.        ,  0.98958333,  0.95833333,  0.90625   ],
                    [ 0.95833333,  0.98958333,  1.        ,  0.98958333,  0.95833333],
                    [ 0.90625   ,  0.95833333,  0.98958333,  1.        ,  0.98958333],
                    [ 0.83333333,  0.90625   ,  0.95833333,  0.98958333,  1.        ]]

        for ii in range(len(X)):
            for jj in range(len(X[0])):
                self.assertAlmostEqual(actual[ii][jj], expected[ii][jj])


### Test the threshold generators
class GenerateThreshold(unittest.TestCase):

    ### 25, 25, 25, 25, 25, 50, 50, 50, 50, 75, 75, 75, 100, 100, 125
    def test_generate_thresholds_SAD(self):
        X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
        expected = [25, 50, 75, 100]
        actual = generate_threshold_values(X, metric='cityblock')
        self.assertEqual(len(expected), len(actual))

        for i in range(len(actual)):
            self.assertEqual(actual[i], expected[i])

if __name__ == '__main__':
    unittest.main()
