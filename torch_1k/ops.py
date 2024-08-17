import numpy as np
from .function import Function


# Square
class Square(Function):

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy

def square(x):
    return Square()(x)


# Exp
class Exp(Function):

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy

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
        return x1 + x2

    def backward(self, gy):
        return gy, gy

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
        x1, x2 = self.inputs[0].data, self.inputs[1].data
        return x2*gy, x1*gy

def mul(x1, x2):
    return Mul()(x1, x2)

# Div
class Div(Function):
    def forward(self, x1, x2):
        return x1 / x2

    def backward(self, gy):
        x1, x2 = self.inputs[0].data, self.inputs[1].data
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
        x = self.inputs[0].data
        c = self.c
        gx = c * x **(c-1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

