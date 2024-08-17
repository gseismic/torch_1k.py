import config
import time
import numpy as np
import torch_1k
from torch_1k import Tensor
from torch_1k import Square, Exp, Add, add, square

torch_1k.log_settings['func_log_enabled'] = False
torch_1k.log_settings['tensor_log_enabled'] = False


def test_autograd_samevar():
    x = Tensor(2.0)

    y = add(x, x)
    y.backward()

    print('x', x)
    assert np.allclose(x.grad, 2)

def test_autograd_multitimes():
    x = Tensor(2.0)

    y = add(x, x)
    y.backward()
    assert np.allclose(x.grad, 2)

    x.zero_grad()
    y = add(x, x)
    y.backward()
    print(x)
    assert np.allclose(x.grad, 2)

def test_autograd_complex_graph():
    x = Tensor(2.0)
    a = square(x)
    # y = (x^2)^2 + (x^2)^2 = 2* x^4
    # y' = 8 * x^3
    y = add(square(a), square(a))
    y.backward()
    # 32, 64
    assert np.allclose(x.grad, 8 * x.data**3)

def test_autograd_memory():
    # from memory_profiler import memory_usage
    def fun():
        for i in range(300):
            x = Tensor(np.random.randn(100000))
            y = square(square(square(x)))
            y.backward()

    # run1
    # t1 - t0 = 1.9104740619659424
    # t3 - t2 = 1.1078150272369385

    # run2
    # t1 - t0 = 1.8910927772521973
    #t3 - t2 = 1.118412971496582
    torch_1k.runtime_settings['remove_recursive_ref'] = False
    t0 = time.time()
    fun()
    t1 = time.time()
    print(f'remove_recursive_ref=False: {t1 - t0 = }')

    # 更快
    torch_1k.runtime_settings['remove_recursive_ref'] = True
    t2 = time.time()
    fun()
    t3 = time.time()
    print(f'remove_recursive_ref=True: {t3 - t2 = }')

    assert t3 - t2 < t1 - t0
    torch_1k.runtime_settings['remove_recursive_ref'] = True
    #mem_usage = memory_usage(func)
    #print(f"Memory usage: {mem_usage}")

def test_autograd_no_grad():
    torch_1k.runtime_settings['remove_recursive_ref'] = True
    def fun():
        for i in range(3):
            x = Tensor(np.random.randn(100000))
            x = square(square(square(x)))
            x = square(square(square(x)))

    # non-no_grad: t1 - t0 = 0.03579282760620117
    # faster: no_grad: t1 - t0 = 0.03249096870422363
    t0 = time.time()
    fun()
    t1 = time.time()
    print(f'non-no_grad: {t1 - t0 = }')

    t0 = time.time()
    with torch_1k.no_grad():
        fun()
    t1 = time.time()
    print(f'no_grad: {t1 - t0 = }')


if __name__ == '__main__':
    if 0:
        test_autograd_samevar()
    if 0:
        test_autograd_multitimes()
    if 0:
        test_autograd_complex_graph()
    if 0:
        test_autograd_memory()
    if 1:
        test_autograd_no_grad()
