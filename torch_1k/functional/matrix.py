import numpy as np
from ..function import Function
from ..utils import ensure_tensor


# Reshape
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        #print(f'{self.shape=}')
        #print('y', y)
        return y

    def backward(self, gy):
        # 反向传播，恢复原始输入的shape
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return ensure_tensor(x)
    # print(f'{x, shape=}')
    return Reshape(shape)(x)


# Transpose
class Transpose(Function):

    #def __init__(self, *dims):
    #    self.dims = dims

    def forward(self, x):
        #if len(self.dims) == 0:
        return np.transpose(x)
        #else:
        #    return np.transpose(x, 

    def backward(self, gy):
        # 反向传播，恢复原始输入的shape
        gx = tranpose(gy)
        return gx

def tranpose(x):
    return Transpose()(x)

