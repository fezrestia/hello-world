from deepzero.Variable import Variable
from deepzero.Parameter import Parameter
from deepzero.Function import add, mul, sub, rsub, div, rdiv, neg, pow
from deepzero.Config import Config, use_config, no_grad
from deepzero.Layer import Layer
from deepzero.Model import Model

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

