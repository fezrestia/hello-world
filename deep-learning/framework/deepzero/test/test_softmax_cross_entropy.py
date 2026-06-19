import unittest
import numpy as np

from ..Variable import Variable
from ..Function import softmax_cross_entropy
from ..utils import gradient_check, array_allclose

class TestSoftmaxCrossEntropy(unittest.TestCase):
    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(int)
        f = lambda x: softmax_cross_entropy(x, Variable(t))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, CLS_NUM = 10, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: softmax_cross_entropy(x, t)
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        N, CLS_NUM = 100, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: softmax_cross_entropy(x, t)
        self.assertTrue(gradient_check(f, x))

