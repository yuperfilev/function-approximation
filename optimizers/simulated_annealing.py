from math import exp
from random import uniform, randint
from functionals.functional import Functional
from functions.abcfunction import Function
from mathtypes.linalg import vector
from optimizers.abcoptimizer import Optimizer

METHOD_ITERATIONS = 100_000
INIT_TEMPERATURE = 1000
EPS = 0.1


class SimulatedAnnealing(Optimizer):
    """Simulated annealing method."""

    def decreaseTemperature(self, temperature: float) -> float:
        t = 1
        while True:
            yield temperature / (1 + t)
            t += 1

    def getTransitionProbability(self, delta: float, temperature: float) -> float:
        return exp(-delta / temperature)

    def isTransition(self, p: float) -> bool:
        return uniform(0, 1) <= p

    def minimize(self, objective: Functional, function: Function, initial_parameters: vector,
                 minimum_parameters: vector, maximum_parameters: vector) -> vector:
        parameters = initial_parameters.copy()
        result_parameters = initial_parameters.copy()
        y1 = objective.value(function.bind(initial_parameters))
        temperature = INIT_TEMPERATURE
        decrease_function = self.decreaseTemperature(temperature)
        k = 0
        while k < METHOD_ITERATIONS and y1 > EPS:
            random_index = randint(0, parameters.length - 1)
            parameters[random_index] = uniform(0, 1) * (maximum_parameters[random_index] -
                                                        minimum_parameters[random_index]) + minimum_parameters[random_index]
            y2 = objective.value(function.bind(parameters))
            if y1 > y2:
                result_parameters = parameters.copy()
                y1 = y2
            else:
                probability = self.getTransitionProbability(
                    y2 - y1, temperature)
                if self.isTransition(probability):
                    result_parameters = parameters.copy()
                    y1 = y2
            temperature = next(decrease_function)
            k += 1
        return result_parameters
