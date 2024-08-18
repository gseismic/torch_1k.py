import config
import time
import numpy as np
import torch_1k
from torch_1k import functional as F
from torch_1k import Tensor, allclose

def test_matrix_basic():
    A = Tensor(np.array([[1.0, 2, 3], [1,2,3]]), "A")
    A2 = A*2
    B = F.sin(A2)
    B.backward(create_graph=True)
    print(A)

def test_matrix_reshape():
    A = Tensor(np.array([[1.0, 2, 3], [1,2,3]]), "A")
    y = F.reshape(A, (6,))
    y.backward(create_graph=True)
    assert allclose(y, [1,2,3,1,2,3])
    print(f'{y=}')
    print(A.grad)
    assert allclose(A.grad, [[1,1,1],[1,1,1]])

def test_matrix_transpose():
    x = Tensor(np.array([[2, 3]]), name='x')
    W = Tensor(np.array([[1], [2]]), name='W')
    #print('x', x)
    #print('x.T', x.T)
    print('transpose x', F.transpose(x))
    assert F.transpose(x).shape == (2, 1)
    # print(W.T)

def test_matrix_mul():
    x = Tensor(np.random.randn(2, 3), name='x')
    W = Tensor(np.random.randn(3, 5), name='W')
    y = F.matmul(x, W)
    # y.backward(retain_grad=True)
    y.backward()
    print(f'{y=}')
    print(f'{x=}')
    print(f'{W=}')
    # assert allclose(A.grad, [[1,1,1],[1,1,1]])

def test_matrix_mul2():
    x = Tensor(np.array([[2, 3]]), name='x')
    W = Tensor(np.array([[1], [2]]), name='W')
    y = F.matmul(x, W)
    # y.backward(retain_grad=True)
    y.backward()
    print(f'{y=}')
    print(f'{x=}')
    print(f'{W=}')
    assert x.grad.shape == (1, 2)
    assert W.grad.shape == (2, 1)

def test_matrix_npsumto():
    # np.broadcast_to 是 NumPy 中的一个函数，用于将一个数组广播（复制）到一个新的形状，
    # 而不实际复制数据。广播的本质是通过调整数组的形状和维度，使其能够适应其他数组的形状，进行元素级操作。
    # np.broadcast_to 就是强制性地将一个数组扩展到指定的形状。

    # 简单广播
    a = np.array([1, 2, 3])
    b = np.broadcast_to(a, (3, 3))
    print(b)

    # 广播标量
    a = np.array(5)
    b = np.broadcast_to(a, (3, 3))
    print(b)

    # 多维数组广播
    a = np.array([[1], [2], [3]])
    b = np.broadcast_to(a, (3, 3))
    print(b)

    a = np.array([1, 2, 3])
    # 将形状 (3,) 的数组广播到 (2, 3) 是合法的
    b = np.broadcast_to(a, (2, 3))
    print(b)
    # 但是将形状 (3,) 的数组广播到 (3, 2) 是不合法的，会引发 ValueError
    # c = np.broadcast_to(a, (3, 2))  # This will raise ValueError

    # 2, 3
    a = np.array([[1,2,3], [10, 20, 30]])
    print('a', a, a.shape)
    y = torch_1k.utils.np_sum_to(a, (2, 1))
    assert allclose(y, [[6], [60]])

    print(y)
    y = torch_1k.utils.np_sum_to(a, (1, 3))
    assert allclose(y, [[11, 22, 33]])
    print(y)


def test_matrix_pad():
    x = Tensor([1], name='x')
    y = F.pad(x, (5,), [2])

    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(y, [0, 0, 1, 0, 0])
    assert allclose(x.grad, [1])

    x = Tensor([1,2], name='x')
    y = F.pad(x, (5,), slice(1,3))
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(y, [0, 1, 2, 0, 0])

    x = Tensor([1,2], name='x')
    y = F.pad(x, (5,), [1,2])
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(y, [0, 1, 2, 0, 0])


def test_matrix_getitem():
    x = Tensor(np.arange(5), name='x')
    y = F.get_item(x, 2)
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(x.grad, [0, 0, 1, 0, 0])
    # print(f'{y=}')

    x = Tensor(np.arange(5), name='x')
    index = slice(1, 3)
    print(f'\n{(index)=}')
    y = F.get_item(x, index)
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(x.grad, [0, 1, 1, 0, 0])

    x = Tensor(np.arange(5), name='x')
    index = np.array([1,2])
    print(f'{(index)=}')
    y = F.get_item(x, index)
    y.backward()
    print(f'{x=}')
    print(f'{y=}')
    assert allclose(x.grad, [0, 1, 1, 0, 0])
    return

def test_matrix_broadcast_to():
    x = Tensor(np.array([[2, 3]]), name='x')
    W = Tensor(np.array([[1], [2]]), name='W')
    y = F.matmul(x, W)
    # y.backward(retain_grad=True)
    y.backward()
    print(f'{y=}')
    print(f'{x=}')
    print(f'{W=}')



if __name__ == '__main__':
    if 0:
        test_matrix_basic()
    if 0:
        test_matrix_reshape()
    if 0:
        test_matrix_transpose()
    if 0:
        test_matrix_mul()
    if 0:
        test_matrix_mul2()
    if 0:
        test_matrix_npsumto()
    if 0:
        test_matrix_pad()
    if 1:
        test_matrix_getitem()
