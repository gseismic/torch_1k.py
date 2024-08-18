from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, parameter, lr=1e-4):
        super().__init__(self, parameters)
        self.lr = lr

    def update_one(self, parameter):
        parameter.data -= self.lr * parameter.grad.data
