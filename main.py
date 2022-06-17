from functionals import l1, l2
from functions import linear_function, polynomial_function, linear_spline, cubic_spline
from mathtypes.linalg import vector
from optimizers import simulated_annealing, gauss, conjugate_gradient


def main():
    # Linear Function, Conjugate Gradient Method
    # x = [vector([2, 1, 3]), vector([3, 2, 3]),
    #      vector([4, 1, 2]), vector([5, 5, 5])]
    # f = vector([12, 17, 17, 31])
    # init_parameters = vector([1, 1, -1, -1])
    # minimum_parameters = vector([-5, -4, -2, -2])
    # maximum_parameters = vector([5, 4, 2, 2])
    # function = linear_function.LinearFunction()
    # functional = l1.L1(x, f)
    # optimizer = conjugate_gradient.ConjugateGradientMethod()
    # result = optimizer.minimize(
    #     functional, function, init_parameters, minimum_parameters, maximum_parameters)
    # print(result)

    #Cubic Spline, Gauss-Newton method
    x = [vector([-1]), vector([1]), vector([2]), vector([3])]
    f = vector([3, 10, 11, -1])
    init_parameters = vector([1, 3, -1 , -1])
    function = cubic_spline.CubicSpline(x)
    functional = l2.L2(x, f)
    optimizer = gauss.GaussNewtonMethod()
    result = optimizer.minimize(functional, function, init_parameters)
    print(result)


if __name__ == "__main__":
    main()
