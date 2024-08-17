import config
import time
import numpy as np
import torch_1k
from torch_1k import Tensor
from torch_1k import Square, Exp, Add, add, square


def test_ops_mul():
    x1 = Tensor(2.0)
    x2 = Tensor(3.0)
    y = x1 * x2
    y.backward()

    assert np.allclose(x1.grad, 3)
    assert np.allclose(x2.grad, 2)

    # x1 = Tensor(2.0)
    # x2 = Tensor(3.0)
    x1.zero_grad()
    y = x1 * x1
    y.backward()

    # print('x1', x1)
    assert np.allclose(x1.grad, 4)

def test_ops_mixup():
    x1 = Tensor(2.0, name='x1')
    y = x1 + 2
    y.backward()
    print('x1', x1)
    assert np.allclose(x1.grad, 1)

    x1 = Tensor(2.0, name='x1')
    y = 2 + x1
    y.backward()
    print('x1', x1)
    assert np.allclose(x1.grad, 1)

    x1 = Tensor(2.0, name='x1')
    y = 2*x1 + 2
    y.backward()
    print('x1', x1)
    assert np.allclose(x1.grad, 2)

    x1 = Tensor(2.0, name='x1')
    y = 2*x1 + np.array(2.5)
    y.backward()
    print('x1', x1)
    assert np.allclose(x1.grad, 2)

    x1 = Tensor(2.0, name='x1')
    y = np.array(2.5) + 2*x1
    y.backward()
    print('x1', x1)
    assert np.allclose(x1.grad, 2)

    x1 = Tensor([2.0], name='x1')
    print(type(x1))
    y = np.array([2.5]) + 2*x1
    print(type(y))
    y.backward()
    print('x1', x1)
    assert np.allclose(x1.grad, 2)

def test_ops_many():
    # neg
    x1 = Tensor(2.0, name="x1")
    y = -x1
    y.backward()
    print(x1)
    assert np.allclose(x1.grad, -1)

    # sub
    x1 = Tensor(2.0, name="x1")
    x2 = Tensor(3.0, name="x2")
    y = 10*x1 - 11*x2
    y.backward()
    print(x1)
    print(x2)
    assert np.allclose(x1.grad, 10)
    assert np.allclose(x2.grad, -11)

    # rsub
    x2 = Tensor(3.0, name="x2")
    y = 10 - 11*x2
    y.backward()
    print(x2)
    assert np.allclose(x2.grad, -11)

    # mul
    x1 = Tensor(2.0, name="x1")
    x2 = Tensor(3.0, name="x2")
    y = x1 * x2
    y.backward()
    assert np.allclose(x1.grad, 3)
    assert np.allclose(x2.grad, 2)

    # div
    x1 = Tensor(2.0, name="x1")
    x2 = Tensor(4.0, name="x2")
    y = x1 / x2
    y.backward()
    assert np.allclose(x1.grad, 1/x2.data)
    assert np.allclose(x2.grad, x1.data*(-1/x2.data**2))

    # rdiv
    x1 = 3
    x2 = Tensor(4.0, name="x2")
    y = x1 / x2
    y.backward()
    assert np.allclose(x2.grad, x1*(-1/x2.data**2))

    # pow
    c = 3
    x2 = Tensor(2.0, name="x2")
    y = x2 ** c
    y.backward()
    print(x2)
    assert np.allclose(x2.grad, c * x2.data **(c-1))


if __name__ == '__main__':
    if 0:
        test_ops_mul()
    if 0:
        test_ops_mixup()
    if 1:
        test_ops_many()
