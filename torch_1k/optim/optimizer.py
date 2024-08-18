
class Optimizer:

    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                self.update_one(parameter)

    def update_one(self, parameter):
        raise NotImplementedError()
