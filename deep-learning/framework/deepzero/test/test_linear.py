import unittest
import numpy as np

from ..Variable import Variable
from ..Function import linear
from ..utils import gradient_check, array_allclose

class TestLinear(unittest.TestCase):
    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        w = Variable(x.data.T)
        b = None
        y = linear(x, w, b)

        res = y.data
        expected = np.array([[14, 32], [32, 77]])
        self.assertTrue(array_allclose(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        W = np.random.randn(2, 3)
        b = np.random.randn(3)
        f = lambda x: linear(x, W, b)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(100, 200)
        W = np.random.randn(200, 300)
        b = None
        f = lambda x: linear(x, W, b)
        self.assertTrue(gradient_check(f, x))

