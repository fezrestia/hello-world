from deepzero.Variable import Variable
from deepzero.Function import add, mul, sub, rsub, div, rdiv, neg, pow
from deepzero.Config import Config, use_config, no_grad

Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__neg__ = neg
Variable.__pow__ = pow

