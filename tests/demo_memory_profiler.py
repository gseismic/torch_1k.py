from memory_profiler import profile


@profile
def example_function():
    data = [i for i in range(10 ** 6)]  # 大量内存使用
    result = [x * 2 for x in data]      # 进一步的内存使用
    del data                           # 删除不再需要的对象
    return result


if __name__ == '__main__':
    from memory_profiler import memory_usage

    def my_function():
        a = [1] * (10 ** 6)
        b = [2] * (2 * 10 ** 7)
        del b
        return a

    mem_usage = memory_usage(my_function)
    print(f"Memory usage: {mem_usage}")
