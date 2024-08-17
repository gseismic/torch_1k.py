from .utils import ensure_ndarray


class Tensor:

    def __init__(self, data):
        self.data = ensure_ndarray(data)
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
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

