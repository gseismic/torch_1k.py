import inspect
import functools
from loguru import logger


# 创建一个装饰器来记录函数调用的日志
def log_function_call(enabled=True):
    '''
    用法:
      self中是否包含log_enabled字段:
        否：是否log由@log_function_call的enabled确定
        是: 同时为真，打log
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # print(f'{log_enabled, self.log_enabled=}')
            should_log = enabled and getattr(self, 'log_enabled', True)
            if should_log:
                logger.info(f"Calling {self.__class__.__name__}.{func.__name__} with args: {args}, kwargs: {kwargs}")

            # 调用原始的函数
            result = func(self, *args, **kwargs)

            if should_log:
                logger.info(f"\t\t{self.__class__.__name__}.{func.__name__} returned: {result}")

            return result
        return wrapper
    return decorator
