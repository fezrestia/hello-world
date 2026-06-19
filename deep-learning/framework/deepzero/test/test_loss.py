import unittest
import numpy as np

from ..Function import mean_squared_error
from ..utils import gradient_check, array_allclose

class TestMeanSquaredError(unittest.TestCase):
    def test_forward1(self):
        x0 = np.array([0.0, 1.0, 2.0])
        x1 = np.array([0.0, 1.0, 2.0])
        expected = ((x0 - x1) ** 2).sum() / x0.size
        y = mean_squared_error(x0, x1)
        self.assertTrue(array_allclose(y.data, expected))

    def test_backward1(self):
        x0 = np.random.rand(10)
        x1 = np.random.rand(10)
        f = lambda x0: mean_squared_error(x0, x1)
        self.assertTrue(gradient_check(f, x0))

    def test_backward2(self):
        x0 = np.random.rand(100)
        x1 = np.random.rand(100)
        f = lambda x0: mean_squared_error(x0, x1)
        self.assertTrue(gradient_check(f, x0))

