import numpy as np
from .tensor import Tensor
from .log import log_function_call
from .settings import log_settings


class Function:

    def __init__(self, log_enabled=None):
        if log_enabled is None:
            self.log_enabled = log_settings.get('func_log_enabled', False)
        else:
            self.log_enabled = log_enabled

    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Tensor(y) for y in ys]
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy


class Exp(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy


class Add(Function):

    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, gy):
        return gy, gy


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x1, x2):
    return Add()(x1, x2)
