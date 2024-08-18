# torch-1k.py
目标：1000行代码以内实现pytorch核心基本功能 Implementing PyTorch's core basic functions within 1000 lines of code

## 核心功能
- [x] Tensor类：要求实现`按元素`的加、减、乘、除
- [x] Tensor类：支持标量和Tensor相加
- [x] Tensor类：支持不同维度下的广播
- [x] 支持函数和复合函数的自动微分求导
- [^] 实现常用函数sin,cos,exp,log,relu,softmax等
- [ ] 实现神经网络Module
- [ ] 实现Linear算子
- [ ] optimizer优化模块: 实现Adam优化算法
- [ ] 实现MLP神经网络，把torch替换为`torch_1k`可以做普通的mnist分类

## 使用说明
例子1:
```
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
```

### 不允许覆盖自身
如下代码错误
```
    x = Tensor(2.0, name="x")
    x = x*x
    x.backward()
```

## 参考资料
- 《深度学习入门自制框架》斋藤康毅 著 郑明智翻

## ChangeLog
- [@2024-08-17] v0.0.1 create project
- [@2024-08-18] v0.0.2
- [@2024-08-19] v0.0.3 实现了基本功能: 核心代码1k, 测试代码1k 
