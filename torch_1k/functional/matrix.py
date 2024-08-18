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
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape) # 反向传播，恢复原始输入的shape

def reshape(x, shape):
    if x.shape == shape:
        return ensure_tensor(x)
    return Reshape(shape)(x)


# Transpose
class Transpose(Function):

    #def __init__(self, *dims):
    #    self.dims = dims

    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, gy):
        gx = transpose(gy) # 反向传播，恢复原始输入的shape
        return gx

def transpose(x):
    return Transpose()(x)


# Unsqueeze: future
class _Unsqueeze(Function):
    # input: 输入的张量。
    # dim: 要插入维度的位置，范围可以是 [-input.dim()-1, input.dim()]。
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        self.x_shape = x.shape
        y = np.expand_dims(x, dim=self.dim)
        return y


# MatMul 
class MatMul(Function):

    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)

#Broadcast
class Broadcast(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        # e.g. (3, 2) <- (3, 1)
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
    # 如果shape不同，则通过sum来缩减dim
    # 如果相同，则不必缩减
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
        if self.axis is not None or self.keepdims:
            assert 0, 'TODO'
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)

def linear(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None # clear
    return y
