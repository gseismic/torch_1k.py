from .log import log_function_call
from .function import Function

from .settings import log_settings, runtime_settings
from .functional import *
from .tensor import (
    Tensor, no_grad, allclose, rand, randn, randint,
    register_ops, manual_seed
)
from .version import __version__

register_ops()
