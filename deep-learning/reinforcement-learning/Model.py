import numpy as np
from typing import override

from deepzero import Variable
from deepzero.Model import Model
from deepzero.Layer import Linear
from deepzero.Function import relu

class QNet(Model):
    def __init__(self) -> None:
        super().__init__()

        self.l1: Linear = Linear(100)
        self.l2: Linear = Linear(4)

    @override
    def forward(self, *inputs: Variable) -> tuple[Variable, ...]:
        x: Variable = inputs[0]

        (x,) = self.l1(x)
        x = relu(x)
        (x,) = self.l2(x)
        return (x,)

