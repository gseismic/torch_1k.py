import numpy as np
from torch_1k.function import Function


# Pad
class Pad(Function):
    '''改变大小，并填充默认数值
    examples:
        x = [6, 6, 2, 3, 9]
        shape = x.shape
        index_or_slices = slice(0,2)
        y = x[index_or_slices] -> [6,6]

        现在由y，返回[6, 6, 0, 0, 0]
    '''
    def __init__(self, shape, index_or_slices, value=0):
        # x.shape, index_or_slices, x[index_or_slices] -> y
        self.shape = shape
        self.index_or_slices = index_or_slices
        self.value = 0

    def forward(self, x):
        '''
        deprecated
        ...
        if isinstance(self.index_or_slices, slice):
            start = self.index_or_slices.start or 0
            if self.index_or_slices.step is not None:
                step = self.index_or_slices.step
            else:
                step = 1
            # stop = self.index_or_slices or 
            n = (self.index_or_slices.stop - start)//step
        '''
        # x.shape 要和self.shape做self.index_or_slices后的大小已知
        y = self.value * np.ones(self.shape)
        y[self.index_or_slices] = x
        return y

    def backward(self, gy):
        gx = gy[self.index_or_slices]
        return gx

def pad(x, shape, index_or_slices, value=0):
    return Pad(shape, index_or_slices, value)(x)
