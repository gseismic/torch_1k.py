import numpy as np
from torch_1k.function import Function
from torch_1k import tensor


class MeanSquaredError(Function):

    def forward(self, x1, x2):
        dif = x1 - x2
        y = (diff ** 2).sum()/len(dif)
        return y

    def backward(self, gy):
        assert 0

def ___(x, index_or_slices):
    assert 0, 'TODO'
