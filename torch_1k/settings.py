import contextlib

log_settings = {
    'func_log_enabled': False,
    'tensor_log_enabled': False
}

runtime_settings = {
    'remove_recursive_ref': True
}

class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    # print('uss....', name, value)
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
