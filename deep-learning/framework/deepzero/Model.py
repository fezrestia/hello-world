from .Variable import Variable
from .Layer import Layer
from .Visualize import plot_dot_graph

class Model(Layer):
    def plot(self, *inputs: Variable, to_file: str = "model.png") -> str:
        (y,) = self.forward(*inputs)
        return plot_dot_graph(y, verbose = True, to_file = to_file)


