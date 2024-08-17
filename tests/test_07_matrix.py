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



if __name__ == '__main__':
    if 0:
        test_matrix_basic()
    if 1:
        test_matrix_reshape()
