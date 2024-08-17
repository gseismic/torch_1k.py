import config
import numpy as np
import torch_1k
from torch_1k import Tensor
from torch_1k import Square, Exp, Add, add
torch_1k.log_settings['func_log_enabled'] = True
torch_1k.log_settings['tensor_log_enabled'] = True


def test_autograd_samevar():
    x = Tensor(2.0)

    y = add(x, x)
    y.backward()

    print('x', x)
    assert np.allclose(x.grad, 2)

def test_autograd_multitimes():
    x = Tensor(2.0)

    y = add(x, x)
    y.backward()
    assert np.allclose(x.grad, 2)

    x.zero_grad()
    y = add(x, x)
    y.backward()
    print(x)
    assert np.allclose(x.grad, 2)


if __name__ == '__main__':
    if 1:
        test_autograd_samevar()
    if 1:
        test_autograd_multitimes()
