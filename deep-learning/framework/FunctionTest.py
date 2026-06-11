import sys
from pathlib import Path
from typing import override
import unittest
import numpy as np

# include same dir layer.
same_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, same_dir)

from Variable import Variable
from Function import square
from Function import numerical_diff

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))  # random input

        y = square(x)
        y.backward()  # analytic differentiation result is in x.grad

        num_grad = numerical_diff(square, x)  # numerical differentiation

        okng = np.allclose(x.grad, num_grad, rtol = 1e-05, atol=1e-08)



# RUN
if __name__ == "__main__":
    unittest.main()

