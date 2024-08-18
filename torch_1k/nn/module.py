import weakref
from .parameter import Parameter


class Module:

    def __init__(self):
        self._parameters = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        self.inputs = [weakref.ref(input) for input in inputs]
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, x):
        raise NotImplementedError()

    def parameters(self):
        for name in self._parameters:
            obj = self.__dict__[name]
            # support Nested-Module
            if isinstance(obj, Module):
                yield from obj.parameters()
            else:
                yield obj

    def zero_grad(self):
        for parameter in self.self.parameters():
            parameter.zero_grad()

