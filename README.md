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
例子2:
[torch的结果](images/torch.png)
[torch-1k的结果](images/torch_1k.png)

```
import matplotlib.pyplot as plt

####################
# 在这里更改参数
use_torch_1k = False
use_torch_1k = True
####################
if use_torch_1k:
    import torch_1k as torch
    import torch_1k.nn as nn
    import torch_1k.optim as optim
    title = 'torch_1k'
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    title = 'torch'

print('#####################################################')
print(f'### Using {title=} ..')
print('#####################################################')
# 创建数据集
torch.manual_seed(0)

# 输入数据 (100个样本)
X = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1)

# 标签数据
true_w = 3
true_b = 2
y = true_w * X + true_b + torch.normal(0, 1, size=X.size())  # 加入少量噪声

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征维度为1，输出维度也为1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
epochs = 1000
losses = []

for epoch in range(epochs):
    model.train()  # 设置模型为训练模式

    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    # 每 100 次迭代打印一次损失
    if (epoch+1) % 50 == 0:
        # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')


# 模型评估
model.eval()
with torch.no_grad():
    predicted = model(X)

# 绘制数据和拟合直线
plt.scatter(X.numpy(), y.numpy(), label='True Data')
plt.plot(X.numpy(), predicted.numpy(), label='Fitted Line', color='r')
plt.title(title)
plt.legend()
#plt.savefig(f"{title}.png")
plt.show()
```

例子2:
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
