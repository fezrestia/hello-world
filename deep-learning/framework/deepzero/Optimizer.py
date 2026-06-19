import numpy as np
from typing import override, Self
from collections.abc import Callable
import math

from .Log import log_d, log_e
from .Parameter import Parameter
from .Model import Model

class Optimizer:
    def __init__(self) -> None:
        self.target: Model|None = None
        self.hooks: list[Callable[[list[Parameter]], None]] = []

    def setup(self, target: Model) -> Self:
        self.target = target
        return self

    def update(self) -> None:
        if self.target is not None:
            params: list[Parameter] = [p for p in self.target.params() if p.grad is not None]
        else:
            log_e(self, "self.target is None")
            assert self.target is not None

        for hook in self.hooks:
            hook(params)

        for param in params:
            self.update_param(param)

    def update_param(self, param: Parameter) -> None:
        raise NotImplementedError()

    def add_hook(self, f: Callable[[list[Parameter]], None]) -> None:
        self.hooks.append(f)


class StochasticGradientDecent(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__()
        self.learning_rate: float = learning_rate

    @override
    def update_param(self, param: Parameter) -> None:
        if param.grad is not None:
            param.data -= self.learning_rate * param.grad.data
        else:
            log_e(self, "param.grad is not None")
            assert param.grad is not None


class MomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__()

        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.id_vs_veloc: dict[int, np.ndarray] = {}

    @override
    def update_param(self, param: Parameter) -> None:
        if param.grad is None:
            log_e(self, "param.grad is None")
            assert param.grad is not None

        veloc_key: int = id(param)

        if veloc_key not in self.id_vs_veloc:
            self.id_vs_veloc[veloc_key] = np.zeros_like(param.data)

        veloc: np.ndarray = self.id_vs_veloc[veloc_key]
        veloc *= self.momentum
        veloc -= self.learning_rate * param.grad.data
        param.data += veloc


class Adam(Optimizer):
    def __init__(self,
            alpha: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t: int = 0
        self.ms: dict[int, np.ndarray] = {}
        self.vs: dict[int, np.ndarray] = {}

    @override
    def update(self) -> None:
        self.t += 1
        super().update()

    @property
    def learning_rate(self) -> float:
        fix1: float = 1.0 - math.pow(self.beta1, self.t)
        fix2: float = 1.0 - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    @override
    def update_param(self, param: Parameter) -> None:
        key: int = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m: np.ndarray = self.ms[key]
        v: np.ndarray = self.vs[key]

        beta1: float = self.beta1
        beta2: float = self.beta2
        eps: float = self.eps
        lr: float= self.learning_rate

        if param.grad is not None:
            grad: np.ndarray = param.grad.data

            m += (1.0 - beta1) * (grad - m)
            v += (1.0 - beta2) * (grad * grad -v)
            param.data -= lr * m / (np.sqrt(v) + eps)
        else:
            log_e(self, "param.grad is None.")
            assert param.grad is not None

