import numpy as np


def ensure_ndarray(data):
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)
