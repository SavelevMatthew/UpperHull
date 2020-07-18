import numpy as np


class TwoDimensions:
    @staticmethod
    def func1(x):
        """
        sin(x)
        """
        return np.sin(4 * x)

    @staticmethod
    def func2(x):
        """
        cos(x^3) * x^2
        """
        return np.cos(x ** 2) * (x ** 3)

    all = [(10, func1.__func__), (10, func2.__func__)]


class ThreeDimensions:
    @staticmethod
    def func1(x, y):
        """
        sin(6x)cos(4y)
        """
        return np.sin(6 * x) * np.cos(4 * y)

    @staticmethod
    def func2(x, y):
        """
        x^2 + y^2
        """
        return x ** 2 + y ** 2

    @staticmethod
    def func3(x, y):
        """
        sin(2y) * y
        """
        return np.sin(y * 2) * y

    @staticmethod
    def func4(x, y):
        """
        cos(2x)cos(y)
        """
        return np.cos(x * 2) * np.cos(y)

    @staticmethod
    def func5(x, y):
        """
        (26 * (x^2 + y^2) - 48xy) / 100
        """
        return (26 * (x ** 2 + y ** 2) - 48 * x * y) / 100

    @staticmethod
    def func6(x, y):
        """
        min(|x|, |y|)
        """
        return min([abs(dim) for dim in [x, y]])

    # all = [(7, func1.__func__), (7, func2.__func__), (7, func3.__func__),
    #        (7, func4.__func__), (7, func5.__func__), (7, func6.__func__)]
    all = [(7, func4.__func__)]
    tracking = [func4.__func__]


class FourDimensions:
    @staticmethod
    def func1(x, y, z):
        """
        (x^2 + y^2 + z^2) / 2
        """
        return (x ** 2 + z ** 2 + y ** 2) / 2

    @staticmethod
    def func2(x, y, z):
        """
        cos(x)sin(y) * z
        """
        return np.cos(x) * np.sin(y) * z

    @staticmethod
    def func3(x, y, z):
        """
        min(|x|, |y|, |z|)
        """
        return min([abs(dim) for dim in [x, y, z]])

    all = [(10, func1.__func__), (10, func2.__func__), (10, func3.__func__)]
