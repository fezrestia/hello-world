import inspect

class Log:
    @staticmethod
    def d(obj: object, msg: str):
        Log.log("D", obj, msg)

    @staticmethod
    def e(obj: object, msg: str):
        Log.log("E", obj, msg)

    @staticmethod
    def log(label: str, obj: object, msg: str):
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

