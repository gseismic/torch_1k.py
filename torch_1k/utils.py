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


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return np.allclose(a.data, b.data, rtol=rtol, atol=atol, equal_nan=equal_nan)

