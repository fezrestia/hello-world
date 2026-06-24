from deepzero.cuda import gpu_enabled
if gpu_enabled:
    print("CUDA on CPU enabled.")
else:
    print("Only on CPU.")

from deepzero.Variable import Variable
from deepzero.Parameter import Parameter
from deepzero.Function import add, mul, sub, rsub, div, rdiv, neg, pow, get_item, img2col, col2img
from deepzero.Config import Config, use_config, no_grad, test_mode
from deepzero.Layer import Layer
from deepzero.Model import Model
from deepzero.Optimizer import Optimizer
from deepzero.DataSet import DataSet
from deepzero.DataLoader import DataLoader, SeqDataLoader
from deepzero.utils import get_conv_outsize

Variable.__add__ = add  # type: ignore[operator, assignment]
Variable.__radd__ = add  # type: ignore[attr-defined, assignment]
Variable.__mul__ = mul  # type: ignore[operator, assignment]
Variable.__rmul__ = mul  # type: ignore[attr-defined, assignment]
Variable.__sub__ = sub  # type: ignore[operator, assignment]
Variable.__rsub__ = rsub  # type: ignore[attr-defined, assignment]
Variable.__truediv__ = div  # type: ignore[operator, assignment]
Variable.__rtruediv__ = rdiv  # type: ignore[attr-defined, assignment]
Variable.__neg__ = neg  # type: ignore[operator, assignment]
Variable.__pow__ = pow  # type: ignore[operator, assignment]
Variable.__getitem__ = get_item  # type: ignore[operator, assignment, misc]

