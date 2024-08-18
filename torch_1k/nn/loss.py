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
        x1, x2 = self.inputs
        gx = pad(gy, self.x_shape, self.index_or_slices, 0)
        return gx
        # equavilent
        #mask = np.zeros(self.x_shape)
        #print(f'{self.y_shape=}')
        #print(f'{self.index_or_slices=}')
        #mask[self.index_or_slices] = 1
        #print(f'{mask=}')
        #pad_gy = pad(gy, self.x_shape, self.index_or_slices, 0)
        #print(f'{gy=}')
        #print(f'{pad_gy=}')
        #gx = tensor.Tensor(mask) * pad_gy
        #return gx

def ___(x, index_or_slices):
    assert 0
    return GetItem(index_or_slices)(x)
