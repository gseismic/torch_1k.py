import config
import numpy as np
import torch_1k
from torch_1k import Tensor
from torch_1k import Square, Exp, Add, add, square
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

def test_autograd_complex_graph():
    x = Tensor(2.0)
    a = square(x)
    # y = (x^2)^2 + (x^2)^2 = 2* x^4
    # y' = 8 * x^3
    y = add(square(a), square(a))
    y.backward()
    # 32, 64
    assert np.allclose(x.grad, 8 * x.data**3)


if __name__ == '__main__':
    if 0:
        test_autograd_samevar()
    if 0:
        test_autograd_multitimes()
    if 1:
        test_autograd_complex_graph()
