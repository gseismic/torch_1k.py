import config
import time
import numpy as np
import torch_1k
from torch_1k import functional as F
from torch_1k import Tensor, allclose

def test_highorder_demo():
    def fun(x):
        return x**4 - 2*x**2

    # f': 4*x**3 - 4*x 
    # f'': 4*3*x**2 - 4 
    x = Tensor(1.0, "x")
    y = fun(x)
    y.backward(create_graph=True)
    print(x)
    assert allclose(x.grad, 4*x**3 - 2*2*x)

    gx = x.grad
    x.zero_grad()
    gx.backward()
    print(x)
    assert allclose(x.grad, 4*3*x**2 - 2*2)

def test_highorder_demo2():
    def fun(x):
        return x**3

    # f': 3*x**2
    # f'': 3*2*x
    x = Tensor(2.0, "x")
    y = fun(x)
    y.backward(create_graph=True)
    print(x)
    assert allclose(x.grad, 3 * x**2)

    gx = x.grad
    x.zero_grad()
    gx.backward()
    print(x)
    assert allclose(x.grad, 3 * 2*x)

def test_highorder_sincos():
    x = Tensor(1.0, "x")
    y = F.sin(x)
    y.backward(create_graph=True)
    print(x)
    for i in range(10):
        gx = x.grad
        x.zero_grad()
        gx.backward(create_graph=True)
        print(x)

def test_highorder_tanh():
    x = Tensor(0, "x")
    y = F.tanh(x)
    y.backward(create_graph=True)
    print(f'{y=}')
    print(x)

if __name__ == '__main__':
    if 0:
        test_highorder_demo()
    if 0:
        test_highorder_demo2()
    if 0:
        test_highorder_sincos()
    if 1:
        test_highorder_tanh()
