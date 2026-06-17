import unittest
import numpy as np

from ..Variable import Variable
from ..Function import sigmoid
from ..utils import gradient_check, array_allclose

class TestSigmoid(unittest.TestCase):
    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y2 = sigmoid(x)
        y = sigmoid(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward2(self):
        x = np.random.randn(10, 10).astype(np.float32)
        y2 = sigmoid(x)
        y = sigmoid(Variable(x))
        res = array_allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        self.assertTrue(gradient_check(sigmoid, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        self.assertTrue(gradient_check(sigmoid, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        self.assertTrue(gradient_check(sigmoid, x_data))

