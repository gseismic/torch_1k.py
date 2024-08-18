import numpy as np
from torch_1k.function import Function
from torch_1k import tensor
from .pad import pad


#GetItem
class GetItem(Function):
    def __init__(self, index_or_slices):
        self.index_or_slices = index_or_slices
        self.x_shape = None

    def forward(self, x):
        # XXX 这样支持多次forward的吗？
        self.x_shape = x.shape
        # 不能forward(self, x, index_or_slices) 设计?
        # get = GetItem()
        # y = get(x, 1), y = get(y, 2)
        # 导致get不知道第一次的index_or_slices(被覆盖了)
        y = x[self.index_or_slices]
        self.y_shape = y.shape
        return y

    def backward(self, gy):
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

def get_item(x, index_or_slices):
    return GetItem(index_or_slices)(x)
