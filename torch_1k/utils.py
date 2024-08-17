import numpy as np


def ensure_ndarray(data):
    # np.isscalar is also OK
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)
