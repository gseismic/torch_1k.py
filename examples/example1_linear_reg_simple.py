import time
import numpy as np
import torch_1k
from torch_1k import functional as F
from torch_1k import Tensor
import matplotlib.pyplot as plt


def run():
    N = 200
    x = np.random.rand(N, 1)
    y_target = 3*x + 1 + 0.3*np.random.rand(N, 1)

    W = Tensor.zeros(1, 1).renamed('W')
    # b = Tensor.zeros(1).renamed('b') NOT-ALLOWED
    b = Tensor.zeros(1, 1).renamed('b')

    def model(x):
        z = F.matmul(x, W).renamed('z')
        y =  z + b
        return y

    def mean_squared_error(predict, target):
        dif = predict - target
        err = F.sum(dif**2) /dif.shape[0]
        return err

    lr = 0.1
    epochs = 1000

    # x = Tensor.zeros(N, 1).renamed('x')
    for i in range(epochs):
        y_pred = model(x)
        loss = mean_squared_error(y_pred, y_target)

        W.zero_grad()
        b.zero_grad()
        loss.backward()
        W.data -= lr*W.grad.data
        b.data -= lr*b.grad.data
        if i % 100 == 0:
            print(f'{i}: loss={loss.data}, {W.data=}, {b.data=}')

    y_pred = model(x)
    plt.scatter(x, y_pred.data, color='g')
    plt.scatter(x, y_target, marker='x')
    plt.show()

if __name__ == '__main__':
    run()
