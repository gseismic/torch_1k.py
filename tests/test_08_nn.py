import config
import time
import numpy as np
import torch_1k
from torch_1k import functional as F
from torch_1k import Tensor, allclose
import torch_1k.nn as nn
from torch_1k.optim import SGD


def test_nn_basic():
    fc = nn.Linear(3, 2, bias=False)
    # x = np.array([[1],[1],[1]]) not allowed
    x = np.array([1,1,1])
    y = fc(x)
    print(f'{fc.weight=}')
    print(f'{fc.bias=}')
    print(f'{x=}')
    print(f'{y=}')

def test_nn_sum():
    x = np.array([1,2])
    y = F.sum(x)
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(y.creator.inputs[0].grad, [1,1])

    x = Tensor(np.array([1,2]))
    y = F.sum(x)
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(x.grad, [1,1])

    x = Tensor(np.array([1,2]))
    y = F.sum(2*x + x)
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(x.grad, [3,3])

    print('='*30)
    x = Tensor(np.array([1,2]))
    print(x**2)
    y = F.sum(x**2)
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(x.grad, [2,4])

def test_nn_linear():
    x = Tensor(np.array([1,2]))
    b = Tensor(1)
    # sum(3*[x1, x2] + [b1, b1])
    # 3*x1 + 3*x2 + 2*b1
    # y'/x1' = 3
    # y'/b' = 2
    y = F.sum(3*x + b)
    print(f'***{b=}')
    y.backward()
    print(f'{x=}')
    print(f'{b=}')
    print(f'{y=}')
    assert allclose(x.grad, [3,3])
    assert allclose(b.grad, 2)

    x = Tensor(np.array([1,1]))
    b = Tensor(0)
    target = Tensor(np.array([1,2]))
    y = F.sum(0.5*(x + b - target)**2)
    print(f'***{b=}')
    y.backward()
    print(f'{x=}')
    print(f'{b=}')
    print(f'{y=}')
    assert allclose(x.grad, [0,-1])
    assert allclose(b.grad, 0-1)

    print('****')
    x = Tensor(np.array([[1], [1]]))
    b = Tensor([[0.0]])
    z = x + b
    print(f'{z=}')
    target = Tensor(np.array([[1],[2]]))
    y = F.sum(0.5*(x + b - target)**2)

    y.backward()
    print(f'{x=}')
    print(f'{b=}')
    print(f'{y=}')
    assert allclose(x.grad, [[0],[-1]])
    assert allclose(b.grad, [[0-1]])
    # assert allclose(b.grad, [0-1])
    # assert allclose(b.grad, 0-1)

def test_nn_linear_regress():
    fc = nn.Linear(3, 2, bias=False)
    np.random.seed(0)

    N = 100
    if 1:
        x = np.random.rand(N, 1)
    else:
        x = np.array([[1], [2], [3]])
    y_target = 3*x + 1 #+ 0.3*np.random.rand(N, 1)
    import matplotlib.pyplot as plt

    W = Tensor.zeros(1, 1).renamed('W')
    # b = Tensor.zeros(1).renamed('b') NOT-ALLOWED
    b = Tensor.zeros(1, 1).renamed('b')

    def model(x):
        z = F.matmul(x, W).renamed('z')
        y =  z + b
        # print(f'{x.shape=}, {W.shape=}, {z.shape=}, {b.shape=}, {y.shape=}')
        # print(f'{x=}, \n{W=}, \n{z=}, \n{b=}')
        # print(f'{y=}')
        return y

    def mean_squared_error(predict, target):
        dif = predict - target
        err = F.sum(dif**2)
        err = F.sum(dif**2) /dif.shape[0]
        return err

    # TODO: 自适应学习率
    lr = 10 # 过大，会导致爆炸
    lr = 0.01
    epochs = 1000

    # x = Tensor.zeros(N, 1).renamed('x')
    for i in range(epochs):
        y_pred = model(x)
        loss = mean_squared_error(y_pred, y_target)

        W.zero_grad()
        b.zero_grad()
        loss.backward()
        #print('-'*50)
        #print(f'{W=}, \n{b=}, \n{loss=}')

        #print(f'{W=}')
        #print(f'{b=}')
        W.data -= lr*W.grad.data
        b.data -= lr*b.grad.data

        if i % 100 == 0:
            print(f'{i}: loss={loss.data}, {W.data=}, {b.data=}')

    y_pred = model(x)
    plt.scatter(x, y_pred.data, color='g')
    plt.scatter(x, y_target, marker='x')
    # plt.show()


if __name__ == '__main__':
    if 0:
        test_nn_basic()
    if 0:
        test_nn_sum()
    if 0:
        test_nn_linear()
    if 1:
        test_nn_linear_regress()
