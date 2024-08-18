import numpy as np
from torch_1k.function import Function
from torch_1k import tensor
from ..misc import mean


class MSELoss(Function):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def forward(self, input, target):
        self.input = input
        self.target = target
        # 所有元素的mean
        return mean((input - target) ** 2)

    def backward(self, grad_output):
        # 从上下文中取出正向传播保存的变量
        input, target = self.input, self.target
        # 计算梯度
        grad_input = (2.0 / np.prod(input.shape)) * (input - target)
        # 返回输入的梯度，target的梯度为None
        return grad_input * grad_output
