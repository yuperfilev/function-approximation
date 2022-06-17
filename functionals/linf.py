from functionals.functional import Functional
from functions.abcfunction import Function
from mathtypes.linalg import vector

class LInf(Functional):
    """
    Linf metrics. For a vector, it is the maximum of its elements.\n
    Does not implement the calculation of the functional gradient.
    
    """
    
    def __init__(self, x: list[vector], f: vector) -> None:
        if isinstance(x, list) and all(isinstance(xi, vector) for xi in x):
            if isinstance(f, vector):
                if f and len(x):
                    if len(x) == f.length:
                        self.x = x.copy()
                        self.f = f.copy()
                    else:
                        raise ValueError(f"len(x) must be equals f.length = {f.length}, but {len(x)}")
                else:
                    raise ValueError("An empty argument(s) was passed")
            else:
                raise TypeError(f"f must be vector not {type(f)}")
        else:
            raise TypeError(f"x must be list of vectors not {type(x)}")

    def value(self, function: Function) -> float:
        max_value = abs(self.f[0] - function.value(self.x[0]))
        for i in range(1, self.f.length):
            value = abs(self.f[i] - function.value(self.x[i]))
            if value > max_value:
                max_value = value
        return max_value