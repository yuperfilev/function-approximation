import unittest
from functionals import l1, l2, linf
from functions import linear_function, polynomial_function
from mathtypes.linalg import vector


class TestL1(unittest.TestCase):

    def setUp(self) -> None:
        self.lf = linear_function.LinearFunction()
        self.pf = polynomial_function.PolynomialFunction()
        self.functional = l1.L1([vector([1.5, 1]), vector(
            [-1, 3]), vector([2, 2])], vector([4, 1, 6]))

    def test_init(self) -> None:
        # В конструктор переданы пустые значения
        with self.assertRaises(ValueError):
            l1.L1([], vector())

        # Не совпадает количество точек и значений искомой функции в этих точках
        with self.assertRaises(ValueError):
            l1.L1([vector([1, 2])], vector([1, 3]))

        # В конструктор передано некорректное значение вместо списка точек
        with self.assertRaises(TypeError):
            l1.L1('a', vector())

        # В конструктор передан не вектор значений искомой функций, а число
        with self.assertRaises(TypeError):
            l1.L1([vector([1, 2])], 1)

    def test_value(self) -> None:
        # Вычисление значения функционала при функции проходящей через все исходный точки
        self.assertEqual(self.functional.value(
            self.lf.bind(vector([2, 1, 0]))), 0)

        # Вычисление значения функционала при функции не проходящей через все исходный точки
        self.assertNotEqual(self.functional.value(
            self.lf.bind(vector([2, 0, 0]))), 0)

        # Вычисление значения функционала при несовпадении области определения функции и пространства исходных точек
        with self.assertRaises(ValueError):
            self.functional.value(self.pf.bind(vector([1, 0, 1])))

    def test_gradient(self) -> None:
        # Вычисление градиента полиномиальной функции, которая не реализует DifferentiableFunction
        with self.assertRaises(TypeError):
            self.functional.gradient(self.pf.bind(vector([2, 1, 0])))

        # Вычисление градиента линейной функции, которая реализует DifferentiableFunction
        self.assertListEqual(list(self.functional.gradient(
            self.lf.bind(vector([1, -1, 0])))), [-2.5, -6, -3])


class TestL2(unittest.TestCase):

    def setUp(self) -> None:
        self.lf = linear_function.LinearFunction()
        self.pf = polynomial_function.PolynomialFunction()
        self.functional = l2.L2([vector([1.5, 1]), vector(
            [-1, 3]), vector([2, 2])], vector([4, 1, 6]))

    def test_init(self) -> None:
        # В конструктор переданы пустые значения
        with self.assertRaises(ValueError):
            l2.L2([], vector())

        # Не совпадает количество точек и значений искомой функции в этих точках
        with self.assertRaises(ValueError):
            l2.L2([vector([1, 2])], vector([1, 3]))

        # В конструктор передано некорректное значение вместо списка точек
        with self.assertRaises(TypeError):
            l2.L2('a', vector())

        # В конструктор передан не вектор значений искомой функций, а число
        with self.assertRaises(TypeError):
            l2.L2([vector([1, 2])], 1)

    def test_value(self) -> None:
        # Вычисление значения функционала при функции проходящей через все исходный точки
        self.assertEqual(self.functional.value(
            self.lf.bind(vector([2, 1, 0]))), 0)

        # Вычисление значения функционала при функции не проходящей через все исходный точки
        self.assertNotEqual(self.functional.value(
            self.lf.bind(vector([2, 0, 0]))), 0)

        # Вычисление значения функционала при несовпадении области определения функции и пространства исходных точек
        with self.assertRaises(ValueError):
            self.functional.value(self.pf.bind(vector([1, 0, 1])))

    def test_gradient(self) -> None:
        # Вычисление градиента полиномиальной функции, которая не реализует DifferentiableFunction
        with self.assertRaises(TypeError):
            self.functional.gradient(self.pf.bind(vector([2, 1, 0])))

        # Вычисление градиента линейной функции, которая реализует DifferentiableFunction
        self.assertListEqual(list(self.functional.gradient(self.lf.bind(vector(
            [1, -1, 0])))), [-8.374602011578231, -20.851049906378453, -9.912794217786477])

    def test_residual(self) -> None:
        # Вычисление остатков, когда функция равна искомой
        self.assertListEqual(list(self.functional.residual(
            self.lf.bind(vector([2, 1, 0])))), [0, 0, 0])

        # Вычисление остатков, когда функция не равна искомой
        self.assertListEqual(list(self.functional.residual(
            self.lf.bind(vector([-2, 1, 0])))), [6.0, -4, 8])

    def test_jacobian(self) -> None:
        # Вычисления якобиана функции, которая не реализует DifferentiableFunction
        with self.assertRaises(TypeError):
            self.functional.jacobian(self.pf.bind(vector([2, 1])))

        # Вычисления якобиана функции, которая реализует DifferentiableFunction
        self.assertListEqual(list(self.functional.jacobian(self.lf.bind(vector([2, 1, 0])))),
                             [[-1.5, -1, -1], [1, -3, -1], [-2, -2, -1]])


class TestLInf(unittest.TestCase):

    def setUp(self) -> None:
        self.lf = linear_function.LinearFunction()
        self.pf = polynomial_function.PolynomialFunction()
        self.functional = linf.LInf([vector([1.5, 1]), vector(
            [-1, 3]), vector([2, 2])], vector([4, 1, 6]))

    def test_init(self) -> None:
        # В конструктор переданы пустые значения
        with self.assertRaises(ValueError):
            linf.LInf([], vector())

        # Не совпадает количество точек и значений искомой функции в этих точках
        with self.assertRaises(ValueError):
            linf.LInf([vector([1, 2])], vector([1, 3]))

        # В конструктор передано некорректное значение вместо списка точек
        with self.assertRaises(TypeError):
            linf.LInf('a', vector())

        # В конструктор передан не вектор значений искомой функций, а число
        with self.assertRaises(TypeError):
            linf.LInf([vector([1, 2])], 1)

    def test_value(self) -> None:
        # Вычисление значения функционала при функции проходящей через все исходный точки
        self.assertEqual(self.functional.value(
            self.lf.bind(vector([2, 1, 0]))), 0)

        # Вычисление значения функционала при функции не проходящей через все исходный точки
        self.assertNotEqual(self.functional.value(
            self.lf.bind(vector([2, 0, 0]))), 0)

        # Вычисление значения функционала при несовпадении области определения функции и пространства исходных точек
        with self.assertRaises(ValueError):
            self.functional.value(self.pf.bind(vector([1, 0, 1])))


if __name__ == "__main__":
    unittest.main()
