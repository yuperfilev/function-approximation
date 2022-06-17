from math import sqrt
from functionals.functional import DifferentiableFunctional, Functional, LeastSquaresFunctional
from functions.abcfunction import DifferentiableFunction, Function
from mathtypes.linalg import vector, matrix

class L2(Functional, DifferentiableFunctional, LeastSquaresFunctional):
    """
    L2 metrics(euclidean norm). For a vector, it is the root of the sum of the squares of all its elements. \n
    Implements the calculation of the gradient of the functional.\n
    Implements the calculation of residuals and Jacobian.

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
            functional_value += (self.f[i] - function.value(self.x[i])) ** 2
        return sqrt(functional_value)

    def gradient(self, function: Function) -> vector:
        if not isinstance(function, DifferentiableFunction):
            raise TypeError("Unable to use this function in L2 functional. The function must implement the abstract class DifferentiableFunction")
        function_gradients = []
        for i in range(len(self.x)):
            function_gradients.append(function.gradient(self.x[i]))
        gradient = vector(function_gradients[0].length)
        y_value = self.value(function)
        for i in range(gradient.length):
            for j in range(len(self.x)):
                delta = self.f[j] - function.value(self.x[j])
                gradient[i] -= (2 * delta * function_gradients[j][i]) / sqrt(y_value)
        return gradient

    def residual(self, function: Function) -> vector:
        residuals = vector()
        for i in range(self.f.length):
            residuals.append(self.f[i] - function.value(self.x[i]))
        return residuals

    def jacobian(self, function: Function) -> matrix:
        if not isinstance(function, DifferentiableFunction):
            raise TypeError("Unable to use this function in L2 functional. The function must implement the abstract class DifferentiableFunction")
        jac_values = [-function.gradient(point) for point in self.x]
        rows = len(jac_values)
        cols = jac_values[0].length
        jac = matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                jac[i, j] = jac_values[i][j]
        return jac
