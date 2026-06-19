import numpy as np
from collections.abc import Callable

from .Variable import Variable
from .Function import Function
from .Function import as_variable
from .Type import Scalar

def gradient_check(
        f: Callable[..., Variable],
        x: np.ndarray|Variable,
        *args: np.ndarray|Variable,
        rtol = 1e-4,  ## relative tolerance
        atol = 1e-5,  ## absolute tolerance
        **kwargs,  # keyward args
):
    x = as_variable(x)
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    if x.grad is not None:
        bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = array_allclose(num_grad, bp_grad, atol = atol, rtol = rtol)

    if not res:
        print(f"# FAILED on gradient_check()")
        print(f"  Numerical Grad")
        print(f"    num_grad.shape = {num_grad.shape}")
        print(f"    num_grad = {num_grad.flatten()[:10]}")
        print(f"  BackProp Grad")
        print(f"    bp_grad.shape = {bp_grad.shape}")
        print(f"    bp_grad = {bp_grad.flatten()[:10]}")
    return res

def numerical_grad(
        f: Callable[..., Variable],
        x: np.ndarray|Variable,
        *args: np.ndarray|Variable,
        **kwargs,
):
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    grad = np.zeros_like(x)

    it = np.nditer(x, flags = ["multi_index"], op_flags = ["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + eps
        y1: np.ndarray|Variable = f(x, *args, **kwargs)  # f(x+h)
        if isinstance(y1, Variable):
            y1 = y1.data
        y1 = y1.copy()

        x[idx] = tmp_val - eps
        y2: np.ndarray|Variable = f(x, *args, **kwargs)  # f(x-h)
        if isinstance(y2, Variable):
            y2 = y2.data
        y2 = y2.copy()

        diff = (y1 - y2).sum()
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_val
        it.iternext()
    return grad

def array_allclose(a, b, rtol = 1e-4, atol = 1e-5):
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    return np.allclose(a, b, atol = atol, rtol = rtol)

def array_equal(a, b):
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    return np.array_equal(a, b)

