import unittest
from functions import linear_function, polynomial_function
from mathtypes.linalg import vector


class TestLinearFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.function = linear_function.LinearFunction()

    def test_value_method_without_parameters(self):
        with self.assertRaises(RuntimeError):
            self.function.value(vector([1, 1, 1]))

    def test_value_method_with_parameters(self):
        self.function.bind(vector([1, -2, 1]))
        self.assertEqual(self.function.value(vector([2, 3])), -3)

    def test_value_method_with_mismatched_point(self):
        self.function.bind(vector([1, -2, 1]))
        with self.assertRaises(IndexError):
            self.function.value(vector([1, 1, 1]))

    def test_gradient(self):
        self.function.bind(vector([1, -2, 1, 2]))
        point = vector([1, 3, -1])
        self.assertListEqual(
            list(self.function.gradient(point)), [1, 3, -1, 1])

    def test_gradient_with_constant_function(self):
        self.function.bind(vector([-2]))
        self.assertListEqual(list(self.function.gradient(vector())), [1])


class TestPolynomialFunction(unittest.TestCase):

    def setUp(self) -> None:
        self.function = polynomial_function.PolynomialFunction()

    def test_value_method_without_parameters(self):
        with self.assertRaises(RuntimeError):
            self.function.value(vector([1, 1, 1]))

    def test_value_method_with_parameters(self):
        self.function.bind(vector([1, -2, 0, 3]))
        self.assertEqual(self.function.value(vector([2])), 3)

    def test_value_method_with_mismatched_point(self):
        self.function.bind(vector([1, -2, 1]))
        with self.assertRaises(ValueError):
            self.function.value(vector([1, 1]))


if __name__ == "__main__":
    unittest.main()
