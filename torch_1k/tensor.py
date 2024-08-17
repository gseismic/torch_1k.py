import numpy as np
from .utils import ensure_ndarray
from .log import log_function_call
from .settings import log_settings


class Tensor:

    def __init__(self, data, log_enabled=None):
        self.data = ensure_ndarray(data)
        if log_enabled is None:
            self.log_enabled = log_settings.get('tensor_log_enabled', False)
        else:
            self.log_enabled = log_enabled
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    @log_function_call(enabled=True)
    def backward(self):
        if self.grad is None:
            # print('one...')
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

    def backward_v1(self):
        # recursive-mode
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

    def shape(self):
        return self.data.shape

    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return (
            f'Tensor({str(self.data)})'
            f'\n with grad={self.grad}'
        )

