import inspect

from .Config import Config

def log_d(obj: object, msg: str):
    if Config.is_debug:
        log_output("D", obj, msg)

def log_e(obj: object, msg: str):
    log_output("E", obj, msg)

def log_output(label: str, obj: object, msg: str):
    clazz = obj.__class__.__name__
    frame = inspect.currentframe()
    func = "N/A"
    line = -1
    if frame is not None:
        caller = frame.f_back
        if caller is not None:
            func = caller.f_code.co_name
            line = caller.f_lineno
    print(f"{label} / {clazz}.{func}:{line} / {msg}")

