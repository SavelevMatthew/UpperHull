import numpy as np
import math
from multiprocessing import Pool, cpu_count
from random import randint
from collections import deque
from timeMeasure import my_timer


@my_timer
def get_upper_convex_hull(builder, G_grid, psi_grid):
    """
    The function make the upper convex hull function of the function psi
    :param G_grid: numpy array of notes of G grid
    :param psi_grid: numpy array of psi values
    :return: numpy array of phi values that is the upper convex hull function
    of the function psi
    """
    dir_grid = builder.get_directions_grid()
    print(len(G_grid))
    args = [(m, G_grid, psi_grid, dir_grid, builder)
            for m in range(len(G_grid))]
    pool = Pool(processes=cpu_count())
    data = pool.map(get_min_unpack, args)
    pool.close()
    return np.array(data)
    # return np.array([get_min(m, G_grid, psi_grid, dir_grid, builder) for m in
    #                  range(len(G_grid))])


def get_min_unpack(packed_args):
    return get_min(*packed_args)


def get_min(m, G_grid, psi_grid, dir_grid, builder):
    alpha = 10
    start_point = randint(0, len(dir_grid) - 1)
    last_point = start_point
    info = "{}/{}".format(m, len(G_grid))
    cache = deque(maxlen=int(2**builder.dim))
    while True:
        grads = list(get_all_inner_grads(m, start_point, alpha, G_grid,
                                         psi_grid, dir_grid))
        new_index = get_step(grads, start_point, last_point, builder)
        if new_index is None:
            print(info + ' Stable exit ' + start_point.__str__())
            return get_body_value(m, start_point, G_grid, psi_grid, dir_grid)
        elif new_index in cache:
            print(info + ' Cycle detected!')
            return min([get_body_value(m, index, G_grid, psi_grid, dir_grid)
                        for index in cache])
        last_point = start_point
        start_point = new_index
        cache.append(new_index)
    # Старая и рабочая версия, разбитая на подфункции
    # return min([get_body_value(m, d, G_grid, psi_grid, dir_grid)
    # for d in range(len(dir_grid))])


def get_step(grads, current_index, last_index, builder):
    pairs = [(i, abs(grads[i])) for i in range(len(grads))]
    ordered = sorted(pairs, key=lambda x: -x[1])
    for move in ordered:
        dim = move[0]
        grad = grads[dim]
        new_index = current_index + math.copysign(builder.ann ** dim, grad)
        new_index = int(new_index)
        if (new_index == last_index or new_index < 0 or
                new_index >= builder.ann ** builder.dim or move[1] < 0.05):
            continue
        # print(new_index, grad)
        return new_index
    return None


def get_max(d, G_grid, psi_grid, dir_grid):
    return max([get_inner_value(d, g, G_grid, psi_grid, dir_grid)
                for g in range(len(G_grid))])


def get_body_value(m, d, G_grid, psi_grid, dir_grid):
    return (get_max(d, G_grid, psi_grid, dir_grid)
            - sum(dir_grid[d] * G_grid[m]))


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
    inner_values = [get_inner_value(d, g, G_grid, psi_grid, dir_grid)
                    for g in range(len(G_grid))]
    # Для каждого значения ищем градиент и домножаем на mki,
    # Что есть mk в одномерном случае
    smooth_grad_values = [smooth_max_grad(i, inner_values, alpha)
                          * G_grid[i][0]
                          for i in range(len(inner_values))]
    # Возвращаем сумму - mi, что просто m в одномерном случае
    return sum([grad for grad in smooth_grad_values]) - G_grid[m][0]


def get_all_inner_grads(m, d, alpha, G_grid, psi_grid, dir_grid):
    """
    :param m: номер точки расчета
    :param d: номер угла поворота
    :param alpha: коэффициент сглаживания
    :param G_grid: сетка значений параметров
    :param psi_grid: сетка значений функции
    :param dir_grid: сетка направлений
    :return: возвращает градиенты по всем координатам,
    аналогично get_inner_grad
    """
    if len(G_grid) == 0:
        return []
    dimensions = G_grid.shape[1]
    inner_values = [get_inner_value(d, g, G_grid, psi_grid, dir_grid) for g in
                    range(len(G_grid))]
    for dim in range(dimensions):
        smooth_grad_values = [
            smooth_max_grad(i, inner_values, alpha) * G_grid[i][dim] for i in
            range(len(inner_values))]
        yield sum([grad for grad in smooth_grad_values]) - G_grid[m][dim]