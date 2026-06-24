import numpy as np
from numpy.typing import DTypeLike
from typing import override
from collections.abc import Callable
from PIL.Image import Image

from .Variable import Variable
from .Function import sigmoid, dropout, relu, pooling, reshape
from .Layer import Layer, Linear, Conv2d, RNN, LSTM
from .Visualize import plot_dot_graph
from .utils import get_file
from .Type import Array

class Model(Layer):
    def plot(self, *inputs: Variable, to_file: str = "model.png") -> str:
        (y,) = self.forward(*inputs)
        return plot_dot_graph(y, verbose = True, to_file = to_file)


class MultiLayerPerceptron(Model):
    def __init__(self,
            ful_con_output_sizes: tuple[int, ...],
            activation: Callable[[Variable], Variable] = sigmoid,
            dropout_ratio: float = 0.5,
    ):
        super().__init__()

        self.activation: Callable[[Variable], Variable] = activation
        self.dropout_ratio: float = dropout_ratio
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
            x = dropout(x, self.dropout_ratio)
            x = self.activation(x)

        (x,) = self.layers[-1](x)
        return (x,)


class VGG16(Model):
    WEIGHTS_PATH = "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz"

    def __init__(self, pretrained: bool = False):
        super().__init__()

        self.conv1_1 = Conv2d(64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv1_2 = Conv2d(64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        self.conv2_1 = Conv2d(128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv2_2 = Conv2d(128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        self.conv3_1 = Conv2d(256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv3_2 = Conv2d(256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv3_3 = Conv2d(256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        self.conv4_1 = Conv2d(512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv4_2 = Conv2d(512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv4_3 = Conv2d(512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        self.conv5_1 = Conv2d(512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv5_2 = Conv2d(512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv5_3 = Conv2d(512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        self.fc6 = Linear(4096)
        self.fc7 = Linear(4096)
        self.fc8 = Linear(1000)

        if pretrained:
            weights_path: str = get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weights_path)

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]

        x = relu(self.conv1_1(x)[0])
        x = relu(self.conv1_2(x)[0])
        x = pooling(x, kernel_size = (2, 2), stride = (2, 2), padding = (0, 0))

        x = relu(self.conv2_1(x)[0])
        x = relu(self.conv2_2(x)[0])
        x = pooling(x, kernel_size = (2, 2), stride = (2, 2), padding = (0, 0))

        x = relu(self.conv3_1(x)[0])
        x = relu(self.conv3_2(x)[0])
        x = relu(self.conv3_3(x)[0])
        x = pooling(x, kernel_size = (2, 2), stride = (2, 2), padding = (0, 0))

        x = relu(self.conv4_1(x)[0])
        x = relu(self.conv4_2(x)[0])
        x = relu(self.conv4_3(x)[0])
        x = pooling(x, kernel_size = (2, 2), stride = (2, 2), padding = (0, 0))

        x = relu(self.conv5_1(x)[0])
        x = relu(self.conv5_2(x)[0])
        x = relu(self.conv5_3(x)[0])
        x = pooling(x, kernel_size = (2, 2), stride = (2, 2), padding = (0, 0))

        x = reshape(x, (x.shape[0], -1))

        x = dropout(relu(self.fc6(x)[0]))
        x = dropout(relu(self.fc7(x)[0]))
        x = self.fc8(x)[0]

        return (x,)

    @staticmethod
    def preprocess(image: Image, size: tuple[int, int] = (224, 224), dtype: DTypeLike = np.float32) -> Image:
        image = image.convert("RGB")
        if size:
            image = image.resize(size)
        img_array: Array = np.asarray(image, dtype = dtype)
        img_array = img_array[:, :, ::-1]  # (H, W, C), invert C (RGB->BGR)
        img_array -= np.array([103.939, 116.779, 123.68], dtype = dtype)
        img_array = img_array.transpose((2, 0, 1))
        return img_array


class SimpleRNN(Model):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()

        self.rnn: RNN = RNN(hidden_size)
        self.fc: Linear = Linear(out_size)

    def reset_state(self) -> None:
        self.rnn.reset_state()

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]

        (h,) = self.rnn(x)
        (y,) = self.fc(h)
        return (y,)


class LSTMRNN(Model):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()

        self.rnn: LSTM = LSTM(hidden_size)
        self.fc: Linear = Linear(out_size)

    def reset_state(self) -> None:
        self.rnn.reset_state()

    @override
    def forward(self, *xs: Variable) -> tuple[Variable, ...]:
        x: Variable = xs[0]

        (y,) = self.rnn(x)
        (y,) = self.fc(y)
        return (y,)

