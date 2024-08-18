import numpy as np

def ensure_ndarray(data):
    # np.isscalar is also OK
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


def np_sum_to(x, shape):
    # e.g. shape(2, 3) -> shape(1, 3)
    # e.g. shape(2, 3) -> shape(2, 1)
    if x.shape == shape:
        return x
    assert len(x.shape) == len(shape)
    num_dif_shapes = 0
    pos_dif_shapes = 0
    for i_dim in range(len(x.shape)):
        if x.shape[i_dim] == shape[i_dim]:
            continue

        assert shape[i_dim] == 1
        num_dif_shapes += 1
        pos_dif_shapes = i_dim

    assert num_dif_shapes == 1, f'SumTo: BadShape: current shape:{x.shape}, target: {shape=}'
    return np.sum(x, axis=pos_dif_shapes, keepdims=True)
