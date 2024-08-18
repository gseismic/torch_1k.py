import config
import time
import numpy as np
import torch_1k
from torch_1k import functional as F
from torch_1k import Tensor, allclose
import torch_1k.nn as nn
from torch_1k.optim import SGD




def test_nn_optimizer():
    fc = nn.Linear(3, 2, bias=False)
    # x = np.array([[1],[1],[1]]) not allowed
    x = np.array([1,1,1])
    y = fc(x)
    print(f'{fc.weight=}')
    print(f'{fc.bias=}')
    print(f'{x=}')
    print(f'{y=}')


if __name__ == '__main__':
    if 1: 
        test_nn_optimizer()
