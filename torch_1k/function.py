import numpy as np
import weakref
from .tensor import Tensor
from .log import log_function_call
from .settings import log_settings, runtime_settings, Config


class Function:

    def __init__(self, log_enabled=None):
        if log_enabled is None:
            self.log_enabled = log_settings.get('func_log_enabled', False)
        else:
            self.log_enabled = log_enabled
        self.generation = None

    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        # inputs 仅仅在反向传播时才需要，不反向传播时，不用保留
        outputs = [Tensor(y) for y in ys]
        # print(Config.enable_backprop)
        if Config.enable_backprop:
            # 更新`代`, 为所有输入代的最大值
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            # 解除输出的循环引用
            if runtime_settings.get('remove_recursive_ref', True):
                self.outputs = [weakref.ref(output) for output in outputs]
            else:
                self.outputs = [output for output in outputs]

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
