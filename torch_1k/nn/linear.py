import numpy as np
from .module import Module
from .parameter import Parameter
from ..functional.matrix import linear


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.random.randn(self.in_features,
                                                self.out_features)*np.sqrt(1/self.in_features),
                                name='W')
        if bias is True:
            self.bias = Parameter(np.zeros(self.out_features), name='b')
        else:
            self.bias = None

    def forward(self, x):
        return linear(x, self.weight, self.bias)

