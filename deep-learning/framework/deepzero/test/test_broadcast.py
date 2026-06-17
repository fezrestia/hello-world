import unittest
import numpy as np

from ..Variable import Variable
from ..Function import sum

class TestBroadcast(unittest.TestCase):
    def test_shape_check(self):
        x = Variable(np.random.randn(1, 10))
        b = Variable(np.random.randn(10))

        #print(f"x.shape = {x.shape}")
        #print(f"b.shape = {b.shape}")

        y = x + b

        #print(f"y.shape = {y.shape}")

        loss = sum(y)

        #print(f"loss.shape = {loss.shape}")

        loss.backward()

        #print(f"x.grad.shape = {x.grad.shape}")
        #print(f"b.grad.shape = {b.grad.shape}")

        self.assertEqual(b.grad.shape, b.shape)

