import unittest
from math import isclose
from optimizers import conjugate_gradient, gauss, simulated_annealing
from functionals import l2, linf
from functions import linear_function
from mathtypes.linalg import vector


class TestConjugateGradientMethod(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizer = conjugate_gradient.ConjugateGradientMethod()
        self.x = [vector([2, 1, 3]), vector([3, 2, 3]),
                vector([4, 1, 2]), vector([5, 5, 5])]
        self.minimum_parameters = vector([-5, -4, -2, -2])
        self.maximum_parameters = vector([5, 4, 2, 2])
        self.f = vector([12, 17, 17, 31])
        self.l2 = l2.L2(self.x, self.f)
    
    def test_minimize(self) -> None:
        # Функционал реализует DifferentiableFunctional.
        # Проверяется близость полученных параметров к истинным с абсолютной погрешностью 0.1.
        function = linear_function.LinearFunction()
        initial_parameters = vector([0, 0, 0, 0])
        true_parameters = vector([3, 2, 1, 1])
        result_parameters = self.optimizer.minimize(self.l2, function, initial_parameters,
            self.minimum_parameters, self.maximum_parameters)
        self.assertEqual(all(isclose(result_parameters[i] - true_parameters[i], 0, abs_tol=0.1) for i in range(result_parameters.length)), True)

        # Функционал не реализует DifferentiableFunctional
        with self.assertRaises(TypeError):
            l_inf = linf.LInf(self.x, self.f)
            self.optimizer.minimize(l_inf, function, initial_parameters, self.minimum_parameters,
                self.maximum_parameters)

class TestGaussNewtonMethod(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizer = gauss.GaussNewtonMethod()
        self.x = [vector([2, 1, 3]), vector([3, 2, 3]),
                vector([4, 1, 2]), vector([5, 5, 5])]
        self.minimum_parameters = vector([-5, -4, -2, -2])
        self.maximum_parameters = vector([5, 4, 2, 2])
        self.f = vector([12, 17, 17, 31])
        self.l2 = l2.L2(self.x, self.f)
    
    def test_minimize(self) -> None:
        # Функционал реализует LeastSquaresFunctional.
        # Проверяется близость полученных параметров к истинным с абсолютной погрешностью 1e-4.
        function = linear_function.LinearFunction()
        initial_parameters = vector([0, 0, 0, 0])
        true_parameters = vector([3, 2, 1, 1])
        result_parameters = self.optimizer.minimize(self.l2, function, initial_parameters,
            self.minimum_parameters, self.maximum_parameters)
        self.assertEqual(all(isclose(result_parameters[i] - true_parameters[i], 0, abs_tol=1e-4) for i in range(result_parameters.length)), True)

        # Функционал не реализует LeastSquaresFunctional
        with self.assertRaises(TypeError):
            l_inf = linf.LInf(self.x, self.f)
            self.optimizer.minimize(l_inf, function, initial_parameters, self.minimum_parameters,
                self.maximum_parameters)

class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizer = simulated_annealing.SimulatedAnnealing()
        self.x = [vector([2, 1, 3]), vector([3, 2, 3]),
                vector([4, 1, 2]), vector([5, 5, 5])]
        self.minimum_parameters = vector([-5, -4, -2, -2])
        self.maximum_parameters = vector([5, 4, 2, 2])
        self.f = vector([12, 17, 17, 31])
        self.l2 = l2.L2(self.x, self.f)
    
    def test_minimize(self) -> None:
        # Проверяется близость полученных параметров к истинным с абсолютной погрешностью 1.
        function = linear_function.LinearFunction()
        initial_parameters = vector([0, 0, 0, 0])
        true_parameters = vector([3, 2, 1, 1])
        result_parameters = self.optimizer.minimize(self.l2, function, initial_parameters,
            self.minimum_parameters, self.maximum_parameters)
        self.assertEqual(all(isclose(result_parameters[i] - true_parameters[i], 0, abs_tol=1) for i in range(result_parameters.length)), True)


if __name__ == "__main__":
    unittest.main()