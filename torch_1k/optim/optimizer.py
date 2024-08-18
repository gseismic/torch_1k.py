import numpy as np


class Optimizer:

    def __init__(self, parameters):
        self.parameters = [item for item in parameters]
        # print([item for item in self.parameters])

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                # parameter.grad = None
                parameter.grad.data.fill(0)

    def step(self):
        # print(f'{self.parameters=}')
        for parameter in self.parameters:
            # print(parameter.grad)
            if parameter.grad is not None:
                self.update_one(parameter)

    def update_one(self, parameter):
        raise NotImplementedError()
