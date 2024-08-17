import numpy as np
from .utils import ensure_ndarray
from .log import log_function_call
from .settings import log_settings, runtime_settings, using_config
from .ops import *


class Tensor:

    # 确保优先级高于np.ndarray的运算符
    __array_priority__ = 200

    def __init__(self, data, name=None, log_enabled=None):
        self.data = ensure_ndarray(data)
        self.name = name
        if log_enabled is None:
            self.log_enabled = log_settings.get('tensor_log_enabled', False)
        else:
            self.log_enabled = log_enabled
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        # Tensor-output代始终比产生它的function大一代
        self.creator = func
        self.generation = self.creator.generation + 1

    def zero_grad(self):
        self.grad = None

    @log_function_call(enabled=True)
    def backward(self, retain_grad=False):
        if self.grad is None:
            # print('one...')
            self.grad = np.ones_like(self.data)

        funcs = []
        set_funcs = set()
        def add_func(f):
            if f not in set_funcs:
                funcs.append(f)
                set_funcs.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop() # pop the item with maximal generation
            # weakref
            # print(runtime_settings)
            if runtime_settings.get('remove_recursive_ref', True):
                gys = [output().grad for output in f.outputs]
            else:
                gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    # in case of: y = x + x
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                # 默认不保留中间导数
                if runtime_settings.get('remove_recursive_ref', True):
                    for output in f.outputs:
                        output().grad = None
                else:
                    for output in f.outputs:
                        output.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        # torch-like, 而不是self.data.size
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return (
            f'Tensor({str(self.data)})[name={self.name}]'
            f'\n with grad={self.grad}'
        )


Tensor.__neg__ = neg

Tensor.__add__ = add
Tensor.__radd__ = add
Tensor.__sub__ = sub
Tensor.__rsub__ = rsub

Tensor.__mul__ = mul
Tensor.__rmul__ = mul
Tensor.__truediv__ = div
Tensor.__rtruediv__ = rdiv

Tensor.__pow__ = pow

def no_grad():
    return using_config('enable_backprop', False)
