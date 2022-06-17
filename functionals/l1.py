from functionals.functional import Functional, DifferentiableFunctional
from functions.abcfunction import Function, DifferentiableFunction
from mathtypes.linalg import vector

class L1(Functional, DifferentiableFunctional):
    """
    L1 metrics. For a vector, it is the sum of the modules of all its elements.\n
    Implements the calculation of the gradient of the functional.

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
        functional_value = 0
        for i in range(self.f.length):
            functional_value += abs(self.f[i] - function.value(self.x[i]))
        return functional_value

    def gradient(self, function: Function) -> vector:
        if not isinstance(function, DifferentiableFunction):
            raise TypeError("Unable to use this function in L1 functional. The function must implement the abstract class DifferentiableFunction")
        function_gradient = []
        for i in range(len(self.x)):
            function_gradient.append(function.gradient(self.x[i]))
        gradient = vector(function_gradient[0].length)
        for i in range(gradient.length):
            for j in range(len(self.x)):
                gradient[i] += sign(function.value(self.x[j]) -
                                    self.f[j]) * function_gradient[j][i]
        return gradient


def sign(value: float) -> int:
    if value >= 0:
        return 1
    return -1