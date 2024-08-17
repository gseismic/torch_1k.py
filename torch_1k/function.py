import numpy as np
import weakref
from . import tensor
# from .tensor import Tensor
from .log import log_function_call
from .settings import log_settings, runtime_settings, Config
from .utils import ensure_tensor


class Function:

    def __init__(self, log_enabled=None):
        if log_enabled is None:
            self.log_enabled = log_settings.get('func_log_enabled', False)
        else:
            self.log_enabled = log_enabled
        self.generation = None

    def __call__(self, *inputs):
        inputs = [ensure_tensor(input) for input in inputs]
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        # `inputs` 仅仅在反向传播时才需要，不反向传播时，不用保留
        outputs = [tensor.Tensor(y) for y in ys]
        # print(Config.enable_backprop)
        if Config.enable_backprop:
            # 更新`代`, 为所有输入代的最大值
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            # 解除输出的循环引用
            if runtime_settings.get('remove_recursive_ref', True):
                self.outputs = [weakref.ref(output) for output in outputs]
            else:
                self.outputs = [output for output in outputs]

        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

