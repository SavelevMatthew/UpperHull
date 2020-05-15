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
    print(m)
    alpha = 0.5
    start_point = randint(0, len(dir_grid) - 1)
    last_point = start_point
    last_grad = 0
    while True:
        grad = get_inner_gradient(m, start_point, alpha, G_grid, psi_grid, dir_grid)
        if abs(grad) < 0.01 or (start_point == 0 and grad > 0) or (start_point == len(dir_grid) - 1 and grad < 0):
            return get_body_value(m, start_point, G_grid, psi_grid, dir_grid)
        elif grad * last_grad < 0:
            return min([get_body_value(m, start_point, G_grid, psi_grid, dir_grid), get_body_value(m, last_point, G_grid, psi_grid, dir_grid)])
        last_point = start_point
        if grad > 0:
            start_point -= 1
        else:
            start_point += 1
        last_grad = grad

    # return min([get_body_value(m, d, G_grid, psi_grid, dir_grid) for d in range(len(dir_grid))])


def get_max(d, G_grid, psi_grid, dir_grid):
    return max([get_inner_value(d, g, G_grid, psi_grid, dir_grid) for g in range(len(G_grid))])


def get_body_value(m, d, G_grid, psi_grid, dir_grid):
    return get_max(d, G_grid, psi_grid, dir_grid) - sum(dir_grid[d] * G_grid[m])


def get_inner_value(d, g, G_grid, psi_grid, dir_grid):
    return psi_grid[g] + sum(dir_grid[d] * G_grid[g])


def get_inner_gradient(m, d, alpha, G_grid, psi_grid, dir_grid):
    max_values = [get_inner_value(d, g, G_grid, psi_grid, dir_grid) for g in range(len(G_grid))]
    smooth_grad = smooth_max_grad(max_values, alpha)
    return sum([smooth_grad[i] * G_grid[i] for i in range(len(smooth_grad))]) - m


# @lru_cache()
def smooth_max(objects, alpha):
    exps = get_exp(objects, alpha)
    numerator = sum([objects[i] * exps[i] for i in range(len(objects))])
    denominator = sum(exps)
    return numerator / denominator


def smooth_max_grad(objects, alpha):
    exps = get_exp(objects, alpha)
    return [(exps[i] / sum(exps)) * (1 + alpha * (objects[i] - smooth_max(objects, alpha))) for i in range(len(objects))]


# @lru_cache()
def get_exp(objects, alpha):
    return [math.e ** (x * alpha) for x in objects]
