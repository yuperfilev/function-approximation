from functions.abcfunction import Function
from mathtypes.linalg import vector


class PolynomialFunction(Function):
    """Polynomial in one-dimensional space. Does not implement gradient calculation."""
    
    def __init__(self) -> None:
        self.parameters = vector()

    def bind(self, parameters: vector) -> Function:
        self.parameters = parameters.copy()
        return self

    def value(self, point: vector) -> float:
        if self.parameters.length != 0:
            if point.length == 1:
                value = 0
                n = self.parameters.length - 1
                for i in range(n):
                    value += self.parameters[i] * point[0] ** (n-i)
                value += self.parameters[-1]
                return value
            else:
                raise ValueError("The number of elements of the vector point must be equal to 1")
        else:
            raise RuntimeError("Function parameters are not set. Use the bind() method to set parameters.")
    
    def __str__(self) -> str:
        if self.parameters.length != 0:
            result_str = "Polynomial function f(x) = "
            for i in range(self.parameters.length - 1):
                result_str += f"{self.parameters[i]}*x^{i} + "
            result_str += f"{self.parameters[-1]}"
            return result_str
        return "Function parameters not set"
