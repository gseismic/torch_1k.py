import numpy as np
from .tensor import Tensor
from . import functional as F


def manual_seed(seed):
    # use numpy random
    np.random.seed(seed)

def unsqueeze(input, dim):
    return Tensor(np.expand_dims(input.data, axis=dim))

def linspace(start, stop, steps):
    data = np.linspace(start, stop, steps)
    return Tensor(data)

def normal(mean=0, std=1.0, size=None):
    data = np.random.normal(loc=mean, scale=std, size=size)
    return Tensor(data)

def mean(x):
    return F.sum(x)/np.prod(x.shape)

