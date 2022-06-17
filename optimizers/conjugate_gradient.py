from functionals.functional import Functional, DifferentiableFunctional
from functions.abcfunction import Function
from mathtypes.linalg import vector
from optimizers.abcoptimizer import Optimizer

GOLDEN_SECTION_MIN = 1e-4
GOLDEN_SECTION_RATIO_1 = 0.381_966_011
GOLDEN_SECTION_RATIO_2 = 0.618_003_398_9
K_MAX_ITERATIONS = 1_000
FUNCIONAL_MIN_VALUE = 1e-3
NORM_MIN_VALUE = 1e-5


class ConjugateGradientMethod(Optimizer):
    """
    Conjugate gradient method.\n
    The minimizing functional must implement the calculation of the gradient.
    
    """

    @classmethod
    def goldenSectionMethod(cls, objective: Functional, function: Function, parameters: vector, gradient: vector):
        a = -5
        b = 5
        lambda1 = a + GOLDEN_SECTION_RATIO_1 * (b - a)
        lambda2 = a + GOLDEN_SECTION_RATIO_2 * (b - a)

        funcValue1 = objective.value(
            function.bind(parameters + lambda1 * gradient))
        funcValue2 = objective.value(
            function.bind(parameters + lambda2 * gradient))

        while b-a > GOLDEN_SECTION_MIN:
            if funcValue1 < funcValue2:
                b = lambda2
                lambda2 = lambda1
                lambda1 = a + GOLDEN_SECTION_RATIO_1 * (b - a)
                funcValue2 = funcValue1
                funcValue1 = objective.value(
                    function.bind(parameters + lambda1 * gradient))
            else:
                a = lambda1
                lambda1 = lambda2
                lambda2 = a + GOLDEN_SECTION_RATIO_2 * (b - a)
                funcValue1 = funcValue2
                funcValue2 = objective.value(
                    function.bind(parameters + lambda2 * gradient))
        return (a + b) / 2

    @classmethod
    def checkBorder(cls, parameters: vector, minimum_parameters: vector, maximum_parameters: vector):
        """Out of bounds is checked, if the value is out of bounds, then it is changed to the boundary value."""
        for i in range(len(parameters)):
            if parameters[i] < minimum_parameters[i]:
                parameters[i] = minimum_parameters[i]
            elif parameters[i] > maximum_parameters[i]:
                parameters[i] = maximum_parameters[i]

    def minimize(self, objective: Functional, function: Function, initial_parameters: vector,
                 minimum_parameters: vector = None, maximum_parameters: vector = None) -> vector:
        if not isinstance(objective, DifferentiableFunctional):
            raise TypeError(
                "Unable to use this functional. The functional must implement the abstract class DifferentiableFunctional")
        result_parameters = initial_parameters.copy()
        while True:
            grad0 = objective.gradient(function.bind(initial_parameters))
            s0 = -grad0
            k = 0
            while True:
                parameters = result_parameters.copy()
                lamb = self.goldenSectionMethod(objective, function, parameters, s0)
                result_parameters = parameters + lamb * s0
                grad1 = objective.gradient(function.bind(result_parameters))
                omega = (grad1.norm() / grad0.norm()) ** 2
                s1 = -grad1 + omega * s0
                s0 = s1.copy()
                grad0 = grad1.copy()
                k += 1
                if objective.value(function) < FUNCIONAL_MIN_VALUE or \
                    (result_parameters - parameters).norm() < NORM_MIN_VALUE or \
                        k > K_MAX_ITERATIONS:
                    break
            if k == K_MAX_ITERATIONS:
                if minimum_parameters is not None and maximum_parameters is not None:
                    self.checkBorder(result_parameters,
                                     minimum_parameters, maximum_parameters)
            else:
                break
        return result_parameters
