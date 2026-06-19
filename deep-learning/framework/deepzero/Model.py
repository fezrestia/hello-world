from typing import override
from collections.abc import Callable

from .Variable import Variable
from .Function import sigmoid
from .Layer import Layer, Linear
from .Visualize import plot_dot_graph

class Model(Layer):
    def plot(self, *inputs: Variable, to_file: str = "model.png") -> str:
        (y,) = self.forward(*inputs)
        return plot_dot_graph(y, verbose = True, to_file = to_file)


class MultiLayerPerceptron(Model):
    def __init__(self,
            ful_con_output_sizes: tuple[int, ...],
            activation: Callable[[Variable], Variable] = sigmoid,
    ):
        super().__init__()

        self.activation: Callable[[Variable], Variable] = activation
        self.layers: list[Layer] = []

        for i, out_size in enumerate(ful_con_output_sizes):
            layer: Layer = Linear(out_size)
            setattr(self, f"l{i}", layer)
            self.layers.append(layer)

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]

        for l in self.layers[:-1]:  # last layer does not run activation.
            (x,) = l(x)
            x = self.activation(x)

        (x,) = self.layers[-1](x)
        return (x,)

