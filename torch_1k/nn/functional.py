import numpy as np
from torch_1k.function import Function


class ReLU(Function):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x)
    return ReLU()(x)
