from abc import ABCMeta, abstractmethod
from mathtypes.linalg import vector

class Function(metaclass = ABCMeta):

    @abstractmethod
    def bind(self, parameters: vector) -> 'Function':
        pass

    @abstractmethod
    def value(self, point: vector) -> float:
        pass


class DifferentiableFunction(metaclass = ABCMeta):

    @abstractmethod
    def gradient(self, point: vector) -> vector:
        pass
