import numpy as np

def ensure_ndarray(data):
    # np.isscalar is also OK
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


def np_sum_to(x, shape):
    '''
    Note:
        shape为目标shape
        根据目标shape确定是否keepdims
    '''
    # e.g. shape(2, 3) -> shape(1, 3)
    # e.g. shape(2, 3) -> shape(2, 1)
    # e.g. shape(3,) -> shape()
    # e.g. shape(3, 3,) -> shape(3, )
    # e.g. shape(3, 5,) -> shape(5, ) not allowed
    # 不同维度时，sum前的所有低位必须对齐
    if x.shape == shape:
        return x

    if len(x.shape) == len(shape):
        has_less_dims = False
    elif len(x.shape)-1 == len(shape):
        has_less_dims = True
    else:
        raise Exception(f'not-allowed: `np_sum_to` {x.shape} -> {shape}')

    # print(f' `np_sum_to` {x.shape} -> {shape}')
    num_dif_shapes = 0
    pos_dif_shapes = -1
    for i_dim in range(len(x.shape)):
        found = False
        if has_less_dims:
            if i_dim >= len(shape):
                # 之前的dim都相同，但x多了1维
                found = True
            elif x.shape[i_dim] != shape[i_dim]:
                # 之前的dim都相同，本dim发现不同
                # 本身就少1维，必须都相同，
                raise Exception(f'not-allowed: `np_sum_to` {x.shape} -> {shape}')
        else:
            if x.shape[i_dim] != shape[i_dim]:
                # 之前的dim都相同，本dim发现不同
                assert shape[i_dim] == 1, f'{i_dim=} of {shape} must be 1'
                # print(f'{i_dim=}, {x.shape[i_dim], shape[i_dim]=}')
                found = True

        if found:
            pos_dif_shapes = i_dim
            num_dif_shapes += 1

    # print(f'{num_dif_shapes, pos_dif_shapes=}')
    assert num_dif_shapes == 1, f'SumTo: BadShape: current shape:{x.shape}, target: {shape=}'

    return np.sum(x, axis=pos_dif_shapes, keepdims=not has_less_dims)
