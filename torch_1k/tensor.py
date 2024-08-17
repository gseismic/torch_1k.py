import numpy as np
from .utils import ensure_ndarray
from .log import log_function_call
from .settings import log_settings, runtime_settings, using_config
from .functional import *
#import networkx as nx
#import matplotlib.pyplot as plt


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

    def plot(self):
        # future: TODO
        import networkx as nx
        import matplotlib.pyplot as plt

        # 创建一个有向无环图
        DAG = nx.DiGraph()

        # 添加节点和边
        DAG.add_edges_from([(1, 2), (1, 3), (3, 4), (4, 5), (2, 5)])

        # 检查是否为DAG
        print("Is DAG:", nx.is_directed_acyclic_graph(DAG))

        # 使用 spring 布局绘制 DAG
        pos = nx.spring_layout(DAG)

        # 绘制节点
        nx.draw_networkx_nodes(DAG, pos, node_size=700, node_color="lightgreen")

        # 绘制边（使用箭头表示方向）
        nx.draw_networkx_edges(DAG, pos, edgelist=DAG.edges(), arrowstyle="->", arrowsize=20)

        # 绘制标签
        nx.draw_networkx_labels(DAG, pos, font_size=20, font_family='sans-serif')

        # 显示图形
        plt.title("Directed Acyclic Graph (DAG)")
        plt.axis('off')  # 关闭坐标轴
        plt.show()

    def zero_grad(self):
        self.grad = None

    @log_function_call(enabled=True)
    def backward(self, retain_grad=False, create_graph=False):
        if self.creator is None:
            # incaseof: x = Tensor(...); x.backward()
            return

        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            self.grad = Tensor(np.ones_like(self.data))

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

            # create_graph默认False: 默认单次backward后不需要再次反向传播了
            # enable_backprop 为了: `inputs` 仅仅在反向传播时才需要，不反向传播时，不用保留
            # 如果create_graph 为真，表示还需要导数值，所以
            # print(create_graph)
            with using_config('enable_backprop', create_graph):
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
            f'Tensor({str(self.data)}, name={self.name}'
            f',grad={self.grad})'
        )


def register_ops():
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


register_ops()

def no_grad():
    return using_config('enable_backprop', False)
