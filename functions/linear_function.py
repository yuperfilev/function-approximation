from functions.abcfunction import DifferentiableFunction, Function
from mathtypes.linalg import vector


class LinearFunction(Function, DifferentiableFunction):
    """Linear n-dimensional function. Implements the calculation of the gradient at a point."""
    
    def __init__(self) -> None:
        self.parameters = vector()

    def bind(self, parameters: vector) -> Function:
        self.parameters = parameters.copy()
        return self

    def value(self, point: vector) -> float:
        if self.parameters.length != 0:
            if point.length == self.parameters.length - 1:
                value = 0
                for i in range(point.length):
                    value += self.parameters[i] * point[i]
                value += self.parameters[-1]
                return value
            else:
                raise IndexError("""The number of elements of the point vector must be equal to
                            the number of function parameters without a free member""")
        else:
            raise RuntimeError("Function parameters are not set. Use the bind() method to set parameters.")

    def gradient(self, point: vector) -> vector:
        """Gradient by parameters of function at a point"""
        
        if self.parameters.length != 0:
            if point.length == self.parameters.length - 1:
                grad = point.copy()
                grad.append(1)
                return grad
            else:
                raise IndexError("""The number of elements of the point vector must be equal to
                                the number of function parameters without a free member""")
        else:
            raise RuntimeError("Function parameters are not set. Use the bind() method to set parameters.")
    
    def __str__(self) -> str:
        if self.parameters.length != 0:
            result_str = "Linear Function f(x) = "
            for i in range(self.parameters.length - 1):
                result_str += f"{self.parameters[i]}*x{i} + "
            result_str += f"{self.parameters[-1]}"
            return result_str
        return "Function parameters not set"