import config
import numpy as np
import torch_1k
from torch_1k import Tensor
from torch_1k import Square, Exp
torch_1k.log_settings['func_log_enabled'] = True
torch_1k.log_settings['tensor_log_enabled'] = True


def test_tensor_basic():
    x = Tensor(np.array(10))
    f = Square()
    y = f(x)
    assert y.data == 10 **2 

def test_tensor_creator():
    A = Square()
    B = Exp()
    C = Exp()

    x = Tensor(0.5)
    a = A(x)
    b = B(a)
    y = C(b)

    print(y)

    assert y.creator is C 
    assert y.creator.input is b
    assert y.creator.input.creator is B
    assert y.creator.input.creator.input.creator is A

def test_backward():
    A = Square(log_enabled=False)
    B = Exp()

    # y = exp(x ** 2)
    x = Tensor(2.0)
    a = A(x)
    y = B(a)

    y.grad = np.array(1.0)
    assert y.creator is B
    assert y.creator.input is a
    assert y.creator.input.creator.input is x

    y.creator.input.grad = y.creator.backward(y.grad)
    y.creator.input.creator.input.grad = y.creator.input.creator.backward(y.creator.input.grad)
    # print(x)
    # 2*x*exp(x**2)
    # print(x.grad, 2*x.data*np.exp(x.data**2))
    assert np.allclose(x.grad, 2*x.data*np.exp(x.data**2))

def test_auto_backward():
    A = Square(log_enabled=True)
    B = Exp()

    # y = exp(x ** 2)
    x = Tensor(2.0)
    a = A(x)
    y = B(a)

    # y.grad = np.array(1.0)
    y.backward()

    print('by auto:', x.grad, 2*x.data*np.exp(x.data**2))
    # |x-y| < atol + rtol * |y|
    assert np.allclose(x.grad, 2*x.data*np.exp(x.data**2), rtol=1e-5,
                       atol=1e-8)


if __name__ == '__main__':
    if 0:
        test_tensor_basic()
    if 0:
        test_tensor_creator()
    if 0:
        test_backward()
    if 1:
        test_auto_backward()
