from functionals.functional import Functional, LeastSquaresFunctional
from functions.abcfunction import Function
from mathtypes.linalg import vector
from optimizers.abcoptimizer import Optimizer


class GaussNewtonMethod(Optimizer):
    """
    Gauss Newton algorithm.\n
    The minimizing functional must implement the calculation of the risiduals and jacobian.
    
    """
    
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
        if not isinstance(objective, LeastSquaresFunctional):
            raise TypeError(
                "Unable to use this functional. The functional must implement the abstract class LeastSquaresFunctional")
        parameters = initial_parameters.copy()
        result_parameters = initial_parameters.copy()
        y1 = objective.value(function.bind(parameters))
        for _ in range(result_parameters.length):
            binded_function = function.bind(parameters)
            jacobian = objective.jacobian(binded_function)
            jac_transposed = jacobian.transpose()
            residuals = objective.residual(binded_function)
            parameters -= jac_transposed.multiply(jacobian).inverse().multiply(
                jac_transposed).multiply(residuals)
            if minimum_parameters and maximum_parameters:
                self.checkBorder(parameters, minimum_parameters, maximum_parameters)
            y2 = objective.value(function.bind(parameters))
            if y1 > y2:
                result_parameters = parameters.copy()
                y1 = y2
        return result_parameters
