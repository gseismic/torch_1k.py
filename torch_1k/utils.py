import numpy as np
from . import tensor


def ensure_ndarray(data):
    # np.isscalar is also OK
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)

def ensure_tensor(data):
    if isinstance(data, tensor.Tensor):
        return data
    return tensor.Tensor(data)
