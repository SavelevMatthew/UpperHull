from measure import Measurer
from multiprocessing import Pool
from random import randint
from collections import deque
from .common import get_body_value, get_inner_value
import numpy as np
import math


@Measurer.timer
def get_upper_convex_hull(threads, builder, g_grid, psi_grid, alpha):
    """
    The function make the upper convex hull function of the function psi
    :param threads: number of available processes to run
    :param builder: builder from grids module
    :param g_grid: numpy array of notes of G grid
    :param psi_grid: numpy array of psi values
    :param alpha: alpha coefficient to use on smooth max
    :return: numpy array of phi values that is the upper convex hull function
    of the function psi
    """
    dir_grid = builder.get_directions_grid()
    # print(len(g_grid))
    args = [(alpha, m, g_grid, psi_grid, dir_grid, builder)
            for m in range(len(g_grid))]
    pool = Pool(processes=threads)
    data = pool.map(unpacked_min, args)
    pool.close()
    return np.array(data)
    # return np.array([get_min(m, G_grid, psi_grid, dir_grid, builder) for m in
    #                  range(len(G_grid))])


def unpacked_min(packed_args):
    return get_min(*packed_args)


def get_min(alpha, m, g_grid, psi_grid, dir_grid, builder):
    current_index = randint(0, len(dir_grid) - 1)
    info = "{}/{}".format(m, len(g_grid))
    cache = deque(maxlen=int(2**builder.dim + 1))
    while True:
        grads = list(get_all_inner_grads(m, current_index, alpha, g_grid,
                                         psi_grid, dir_grid))
        new_index = get_step(grads, current_index, builder)
        if new_index is None:
            # print(info + ' Stable exit ' + str(current_index))
            return get_body_value(m, current_index, g_grid, psi_grid, dir_grid)
        elif new_index in cache:
            # print(info + ' Cycle detected!')
            return min([get_body_value(m, index, g_grid, psi_grid, dir_grid)
                        for index in cache])
        current_index = new_index
        cache.append(new_index)


def get_step(grads, current_index, builder):
    pairs = [(i, abs(grads[i])) for i in range(len(grads))]
    ordered = sorted(pairs, key=lambda x: -x[1])
    for move in ordered:
        dim = move[0]
        grad = grads[dim]
        new_index = current_index + math.copysign(builder.ann ** dim, grad)
        new_index = int(new_index)
        if (new_index < 0 or
                new_index >= builder.ann ** builder.dim or move[1] < 0.05):
            continue
        # print(new_index, grad)
        return new_index
    return None


def get_all_inner_grads(m, d, alpha, g_grid, psi_grid, dir_grid):
    if len(g_grid) == 0:
        return []
    dimensions = g_grid.shape[1]
    inner_values = [get_inner_value(d, g, g_grid, psi_grid, dir_grid) for g in
                    range(len(g_grid))]
    for dim in range(dimensions):
        smooth_grad_values = [
            smooth_max_grad(i, inner_values, alpha) * g_grid[i][dim] for i in
            range(len(inner_values))]
        yield sum([grad for grad in smooth_grad_values]) - g_grid[m][dim]


def smooth_max_grad(index, objects, alpha):
    exps = get_exponents(objects, alpha)
    sm_max = smooth_max(objects, exps)
    return (exps[index] / sum(exps)) * (1 + alpha * (objects[index] - sm_max))


def get_exponents(objects, alpha):
    return [math.e ** (alpha * obj) for obj in objects]


def smooth_max(objects, exponents):
    numerator = sum([objects[i] * exponents[i] for i in range(len(objects))])
    denominator = sum(exponents)
    return numerator / denominator
