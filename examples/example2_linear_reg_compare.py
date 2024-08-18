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
