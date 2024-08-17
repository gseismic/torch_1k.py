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

    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Tensor(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    @log_function_call(enabled=True)
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):

    @log_function_call(enabled=True)
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy


class Exp(Function):

    @log_function_call(enabled=True)
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy
