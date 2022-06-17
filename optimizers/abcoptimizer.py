from abc import ABCMeta, abstractmethod
from functionals.functional import Functional
from functions.abcfunction import Function
from mathtypes.linalg import vector

class Optimizer(metaclass=ABCMeta):
    
    @abstractmethod
    def minimize(objective: Functional, function: Function, initial_parameters: vector,
        minimum_parameters: vector=None, maximum_parameters: vector=None) -> vector:
        pass