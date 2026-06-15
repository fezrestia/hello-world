import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def use_config(name: str, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return use_config("enable_backprop", False)

