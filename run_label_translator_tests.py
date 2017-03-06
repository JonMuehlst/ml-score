#!/usr/bin/env python

from translate_labels import *
import numpy as np
import pandas as pd
import unittest
import nose

class TestLabelsTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y = [ 13504, 12403, 11302 ]
        cls.df = pd.DataFrame(cls.y, columns=['labels'])

    def test_browser_app(self):
        np.testing.assert_array_equal(transform_to_browser_app_labels(self.y), np.array([3500, 2400, 1300]))

    def test_os_app(self):
        np.testing.assert_array_equal(transform_to_os_app_labels(self.y), np.array([3004, 2003, 1002]))

    def test_os_browser(self):
        np.testing.assert_array_equal(transform_to_os_browser_labels(self.y), np.array([504, 403, 302]))

    def test_os(self):
        np.testing.assert_array_equal(transform_to_os_labels(self.y), np.array([4, 3, 2]))

    def test_browser(self):
        np.testing.assert_array_equal(transform_to_browser_labels(self.y), np.array([500, 400, 300]))

    def test_app(self):
        np.testing.assert_array_equal(transform_to_app_labels(self.y), np.array([3000, 2000, 1000]))
        np.testing.assert_array_equal(transform_to_app_labels(self.df['labels']), np.array([3000, 2000, 1000]))


if __name__ == "__main__":
    nose.main()
