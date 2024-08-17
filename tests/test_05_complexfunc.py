import config
import time
import numpy as np
import torch_1k
from torch_1k import Tensor, allclose
from torch_1k import Square, Exp, Add, add, square


def test_complexfun_self():
    x = Tensor(2.0, name="x")
    y = x
    y.backward()
    print(x)
    assert x.grad is None

    x = Tensor(2.0, name="x")
    x = x*x
    x.backward()
    print(x)
    assert x.grad is None

    # XXX not-allowed
    #x = Tensor(2.0, name="x")
    #x = x*x # NOT-allowed
    #print('---', x)
    #x.zero_grad()
    #y = x*2
    #y.backward()
    #print(x)
    #assert x.grad is None
    # x = x*2


def test_complexfun_userdefine():
    def fun(x):
        return x*1 + x**2

    x1 = Tensor(1.0, "x")
    y = fun(x1)
    y.backward()
    print(x1)
    assert allclose(x1.grad, 3)

    def fun(x):
        t1 = 2*x + 5
        t2 = 3*t1 + 2*x
        return t2

    x1 = Tensor(1.0, "x")
    y = fun(x1)
    y.backward()
    print(x1)
    assert allclose(x1.grad, 8)


def test_complexfun_userdefine2():
    # future: x = x + x
    def fun(x):
        return x*1 + x**2

    x1 = Tensor(1.0, "x")
    y = fun(x1)
    y.backward()
    print(x1)
    assert allclose(x1.grad, 3)

if __name__ == '__main__':
    if 0:
        test_complexfun_self()
    if 0:
        test_complexfun_userdefine()
    if 1:
        test_complexfun_userdefine2()
