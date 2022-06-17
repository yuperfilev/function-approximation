from functions.abcfunction import Function, DifferentiableFunction
from mathtypes.linalg import vector

class LinearSpline(Function, DifferentiableFunction):
    """
    Linear spline for one-dimensional space. Implements the calculation of the gradient at a point.
    Nodes and function values (parameters) are passed to them in ascending order of node value.
    
    """

    def __init__(self, nodes: list) -> None:
        self.nodes = list(map(float, nodes))

    def bind(self, parameters: vector) -> Function:
        self.parametes = parameters.copy()
        return self

    def basicFunction(self, point: float, num_func: int, index: int) -> float:
        """Returns the value of the basis function at point."""
        h = self.nodes[index + 1] - self.nodes[index]
        if num_func == 0:
            return 1 - (point - self.nodes[index]) / h
        else:
            return (point - self.nodes[index]) / h

    def value(self, point: vector) -> float:
        cur_point = float(point)
        index = self.getIntervalNumber(cur_point)
        value = self.parametes[index] * self.basicFunction(cur_point, 0, index) + \
            self.parametes[index + 1] * self.basicFunction(cur_point, 1, index)
        return value

    def gradient(self, point: vector) -> vector:
        cur_point = float(point)
        gradient = vector(self.parametes.length)
        index = self.getIntervalNumber(cur_point)
        gradient[index] = self.basicFunction(cur_point, 0, index)
        gradient[index + 1] = self.basicFunction(cur_point, 1, index)
        return gradient

    def getIntervalNumber(self, point: float) -> int:
        """Returns the index of the left node of the interval in which point falls"""
        for index in range(self.nodes.length - 1):
            if point <= self.nodes[index + 1]:
                break
        return index
