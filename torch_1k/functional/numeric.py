import numpy as np
from ..function import Function
from .matrix import sum_to


# Square
class Square(Function):

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        return 2 * x * gy

def square(x):
    return Square()(x)


# Exp
class Exp(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        # 为了计算高阶导数, 要求grad也为Tensor类型
        # x = self.inputs[0].data
        x = self.inputs[0]
        # return np.exp(x) * gy
        return exp(x) * gy

def exp(x):
    return Exp()(x)

# Neg
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

# Add
class Add(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        # 发生了隐式broadcast -> sum
        #print('---add---')
        #print('x1', repr(x1))
        #print('x2', repr(x2))
        #print(f'{x1.shape=}, {x2.shape=}')
        y = x1 + x2
        # self.broadcast_shape = y.shape
        return y

    def backward(self, gy):
        gx1, gx2 = gy, gy
        #print(f'***{self.x1_shape=}, {self.x2_shape=}, {self.broadcast_shape=}')
        #if self.x1_shape != self.broadcast_shape:
        #    gx1 = sum_to(gx1, self.x1_shape)
        #if self.x2_shape != self.broadcast_shape:
        #    # 梯度回到自己原来的shape
        #    gx2 = sum_to(gx2, self.x2_shape)
        # 与上面等价
        if self.x1_shape != self.x2_shape:
            x1, x2 = self.inputs
            # print(f'sum----to: {self.x1_shape=}, {self.x2_shape=}')
            # print(f'{x1=},\n {x2=}')
            gx1 = sum_to(gx1, self.x1_shape)
            gx2 = sum_to(gx2, self.x2_shape)
        return gx1, gx2

def add(x1, x2):
    return Add()(x1, x2)


# Sub
class Sub(Function):
    def forward(self, x1, x2):
        return x1 - x2

    def backward(self, gy):
        return gy, -gy

def sub(x1, x2):
    return Sub()(x1, x2)

def rsub(x1, x2):
    # swap
    return Sub()(x2, x1)


# Mul
class Mul(Function):
    def forward(self, x1, x2):
        return x1 * x2

    def backward(self, gy):
        x1, x2 = self.inputs[0], self.inputs[1]
        return x2*gy, x1*gy

def mul(x1, x2):
    return Mul()(x1, x2)

# Div
class Div(Function):
    def forward(self, x1, x2):
        return x1 / x2

    def backward(self, gy):
        x1, x2 = self.inputs[0], self.inputs[1]
        return gy/x2, -gy*x1 / x2 ** 2

def div(x1, x2):
    return Div()(x1, x2)

def rdiv(x1, x2):
    # swap
    return Div()(x2, x1)

# Pow
class Pow(Function):
    def __init__(self, c, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x **(c-1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

# Sin
class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0]
        # gx = gy * np.cos(x)
        gx = gy * cos(x) # call Cos()(x)
        return gx

def sin(x):
    return Sin()(x)

# Cos
class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x = self.inputs[0]
        # gx = gy * np.sin(x)
        gx = - gy * sin(x)
        return gx

def cos(x):
    return Cos()(x)

# Tanh
class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        # 1 - y**2
        y = self.outputs[0]()
        gx = gy * (1 - y*y)
        return gx

def tanh(x):
    return Tanh()(x)

# Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        return 1/(1+np.exp(-x))

    def backward(self, gy):
        return gy * np.exp(-x) / (1+np.exp(-x))**2

def sigmoid(x):
    return Sigmoid()(x)
