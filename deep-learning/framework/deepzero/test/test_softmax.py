import unittest
import numpy as np

from ..Function import softmax
from ..utils import gradient_check, array_allclose

class TestSoftmax(unittest.TestCase):
    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

