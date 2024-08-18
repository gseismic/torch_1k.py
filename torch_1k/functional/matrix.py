import numpy as np
from torch_1k.function import Function
from torch_1k.tensor import Tensor, ensure_tensor
from torch_1k.utils import np_sum_to


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
        #print('----forward---')
        #print('x', x, repr(x))
        y = np.transpose(x)
        #print('y', y, repr(y))
        #print('x', x, repr(x))
        return y
        #else:
        #    return np.transpose(x, 

    def backward(self, gy):
        # 反向传播，恢复原始输入的shape
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)


# MatMul 
class MatMul(Function):

    def forward(self, x, W):
        #print('---forward---')
        #print('x', x)
        #print('W', W)
        y = x.dot(W)
        #print('y', y)
        return y

    def backward(self, gy):
        #print('---backward---')
        #print('gy', gy)
        x, W = self.inputs
        #print('x', x)
        #print('W', W)
        gx = matmul(gy, W.T)
        #print(f'{gx=}')
        gW = matmul(x.T, gy)
        #print(f'{gW=}')
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)

#Broadcast
class Broadcast(Function):
    def __init__(self, shape):
        # e.g. (3, 2)
        self.shape = shape

    def forward(self, x):
        # e.g. (3, 1)
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        # 沿着扩展后的x_shape再加回来
        # XXX NOTE 怀疑有问题，这对sum(y) 这样椒可以的，但如果是get_item呢？y[1] 
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return ensure_tensor(x)
    return Broadcast(shape)(x)


#SumTo
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np_sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return ensure_tensor(x)
    return SumTo(shape)(x)

#Sum
class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):

        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)
