import numpy as np
import math
from functools import lru_cache
from random import randint
from grids import get_directions_grid


def get_upper_convex_hull(G_grid, psi_grid, dir_grid_ann):
    """
    The function make the upper convex hull function of the function psi
    :param G_grid: numpy array of notes of G grid
    :param psi_grid: numpy array of psi values
    :param dir_grid_ann: axis notes number
    :return: numpy array of phi values that is the upper convex hull function of the function psi
    """
    dim = G_grid.shape[1]
    dir_grid = get_directions_grid(dim, dir_grid_ann)
    print(len(G_grid))
    return np.array([get_min(m, G_grid, psi_grid, dir_grid) for m in range(len(G_grid))])


def get_min(m, G_grid, psi_grid, dir_grid):
    alpha = 0.5
    start_point = randint(0, len(dir_grid) - 1)
    last_point = start_point
    last_grad = 0
    info = "{}/{}".format(m, len(G_grid))
    while True:
        grad = get_inner_grad(m, start_point, alpha, G_grid, psi_grid, dir_grid)
        if (start_point == 0 and grad > 0) or (start_point == len(dir_grid) - 1 and grad < 0):
            print(info + ' ex1 ' + start_point.__str__())
            return get_body_value(m, start_point, G_grid, psi_grid, dir_grid)
        if grad * last_grad < 0:
            print(info + ' ex2')
            return min([get_body_value(m, d, G_grid, psi_grid, dir_grid) for d in [start_point, last_point]])
        last_point = start_point
        last_grad = grad
        if grad > 0:
            start_point -= 1
        else:
            start_point += 1
    # Старая и рабочая версия, разбитая на подфункции
    # return min([get_body_value(m, d, G_grid, psi_grid, dir_grid) for d in range(len(dir_grid))])


def get_max(d, G_grid, psi_grid, dir_grid):
    return max([get_inner_value(d, g, G_grid, psi_grid, dir_grid) for g in range(len(G_grid))])


def get_body_value(m, d, G_grid, psi_grid, dir_grid):
    return get_max(d, G_grid, psi_grid, dir_grid) - sum(dir_grid[d] * G_grid[m])


def get_inner_value(d, g, G_grid, psi_grid, dir_grid):
    return psi_grid[g] + sum(dir_grid[d] * G_grid[g])

# Smooth max region


def get_exponents(objects, alpha):
    """
    Получаем экспоненты (в будущем добавить кэш, что не считать кучу раз)
    :param objects: числовое множество
    :param alpha: коэффициент сглаживания
    :return: экспоненты
    """
    return [math.e ** (alpha * obj) for obj in objects]


def smooth_max(objects, exponents):
    """
    Расчет плавного максимума согласно формулы из вики
    :param objects: числовое множество
    :param exponents: коэффициент сглаживания
    :return: сглаженный максимум
    """
    numerator = sum([objects[i] * exponents[i] for i in range(len(objects))])
    denominator = sum(exponents)
    return numerator / denominator


def smooth_max_grad(index, objects, alpha):
    """
    Получает значение градиента по элементу с номером index
    :param index: индекс получаемого градиента
    :param objects: числовое множество
    :param alpha: коэффициент сглаживания
    :return: значение градиента
    """
    exps = get_exponents(objects, alpha)
    sm_max = smooth_max(objects, exps)
    return (exps[index] / sum(exps)) * (1 + alpha * (objects[index] - sm_max))

# grad region


def get_inner_grad(m, d, alpha, G_grid, psi_grid, dir_grid):
    # Получили множество значений внутренних скобок
    # все, что внутри max в оригинальной формуле
    inner_values = [get_inner_value(d, g, G_grid, psi_grid, dir_grid) for g in range(len(G_grid))]
    # Для каждого значения ищем градиент и домножаем на mki,
    # Что есть mk в одномерном случае
    smooth_grad_values = [smooth_max_grad(i, inner_values, alpha) * G_grid[i] for i in range(len(inner_values))]
    # Возвращаем сумму - mi, что просто m в одномерном случае
    return sum([grad for grad in smooth_grad_values]) - G_grid[m]