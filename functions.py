import numpy as np


class ThreeDimensions:
    @staticmethod
    def func1(x, y):
        return np.sin(6 * x) * np.cos(4 * y)

    @staticmethod
    def func2(x, y):
        return x ** 2 + y ** 2

    @staticmethod
    def func3(x, y):
        return np.sin(y * 2) * y

    @staticmethod
    def func4(x, y):
        return np.cos(x * 2) * np.cos(y)

    @staticmethod
    def func5(x, y):
        return (0.26 * ((10 * x) ** 2 + (10 * y) ** 2) - 48 * x * y) / 100

    all = [(10, func1.__func__), (10, func2.__func__), (10, func3.__func__),
           (10, func4.__func__), (10, func5.__func__)]
