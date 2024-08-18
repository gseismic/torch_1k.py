import config
import time
import numpy as np
import torch_1k
import torch
from torch_1k import functional as F
from torch_1k import Tensor, allclose
import torch_1k.nn as nn
from torch_1k.optim import SGD


def test_func_basic():
    A = torch_1k.linspace(0, 1, 11)
    A_torch = torch.linspace(0, 1, 11)
    print(repr(A.numpy()))
    print(repr(A_torch.numpy()))
    assert np.allclose(A.numpy(), A_torch.numpy())

    A = torch_1k.unsqueeze(torch_1k.linspace(0, 1, 11), dim=0)
    A_torch = torch.unsqueeze(torch.linspace(0, 1, 11), dim=0)
    print(f'{A=}')
    print(f'{A_torch=}')
    assert np.allclose(A.numpy(), A_torch.numpy())

    A = torch_1k.normal(0, 1, size=(3, 5))
    A_torch = torch.normal(0, 1, size=(3, 5))
    print(f'{A=}')
    print(f'{A_torch=}')
    assert A.numpy().shape == A_torch.numpy().shape


if __name__ == '__main__':
    if 1: 
        test_func_basic()
