from .optimizer import Optimizer


class MomentumSGD(Optimizer):

    def __init__(self, parameter, lr=1e-4, momentum=0.9):
        super().__init__(self, parameters)
        self.lr = lr
        self.momentum = momentum

    def update_one(self, parameter):
        assert 0
        parameter.data -= self.lr * parameter.grad.data

