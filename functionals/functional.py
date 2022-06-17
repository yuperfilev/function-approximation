from abc import ABCMeta, abstractmethod
from functions.abcfunction import Function
from mathtypes.linalg import vector, matrix

class Functional(metaclass = ABCMeta):

    @abstractmethod
    def value(self, function: Function) -> float:
        pass

class DifferentiableFunctional(metaclass = ABCMeta):

    @abstractmethod
    def gradient(self, function: Function) -> vector:
        pass

class LeastSquaresFunctional(metaclass = ABCMeta):

    @abstractmethod
    def residual(self, function: Function) -> vector:
        pass

    @abstractmethod
    def jacobian(self, function: Function) -> matrix:
        pass
