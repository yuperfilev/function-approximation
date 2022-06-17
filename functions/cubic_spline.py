from functions.abcfunction import Function, DifferentiableFunction
from mathtypes.linalg import vector

class CubicSpline(Function, DifferentiableFunction):
    """
    Cubic spline for one-dimensional space. Implements the calculation of the gradient at a point.
    Nodes and function values (parameters) are passed to them in ascending order of node value.

    """

    def __init__(self, nodes: list):       
        self.nodes = list(map(float, nodes))

    def bind(self, parameters: vector) -> Function:
        self.parameters = parameters.copy()
        self.derivative_parameters = vector(self.parameters.length)
        self.derivative_parameters[0] = 0.5 * (
            self.parameters[1] - self.parameters[0]) / (self.nodes[1] - self.nodes[0])
        return self

    def basicFunction(self, point: float, numFunc: int, index: int) -> float:
        """Returns the value of the basis function at point."""
        h = self.nodes[index + 1] - self.nodes[index]
        ksi = (point - self.nodes[index]) / h
        if numFunc == 0:
            return 1 - 3 * ksi ** 2 + 2 * ksi ** 3
        if numFunc == 1:
            return h * (ksi - 2 * ksi ** 2 + ksi ** 3)
        if numFunc == 2:
            return 3 * ksi ** 2 - 2 * ksi ** 3
        else:
            return h * (-ksi ** 2 + ksi ** 3)

    def getIntervalNumber(self, point: float) -> int:
        """Returns the index of the left node of the interval in which point falls"""
        for index in range(len(self.nodes) - 1):
            if point <= self.nodes[index + 1]:
                break
        return index

    def value(self, point: vector) -> float:
        cur_point = float(point)
        index = self.getIntervalNumber(cur_point)
        value = self.parameters[index] * self.basicFunction(cur_point, 0, index) + \
            self.derivative_parameters[index + 1] * self.basicFunction(cur_point, 1, index) + \
            self.parameters[index + 1] * self.basicFunction(cur_point, 2, index) + \
            self.derivative_parameters[index + 1] * \
            self.basicFunction(cur_point, 3, index)
        return value

    def gradient(self, point: vector) -> vector:
        cur_point = float(point)
        grad = vector(self.parameters.length)
        index = self.getIntervalNumber(cur_point)
        grad[index] = self.basicFunction(cur_point, 0, index)
        grad[index + 1] = self.basicFunction(cur_point, 2, index)
        return grad
