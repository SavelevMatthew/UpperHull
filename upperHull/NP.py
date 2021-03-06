from measure import Measurer
from multiprocessing import Pool
from .common import get_body_value
import numpy as np


@Measurer.timer
def get_upper_convex_hull(threads, builder, g_grid, psi_grid, should_track=False):
    """
    :param threads: number of allowed processes
    :param builder: builder from grids module
    :param g_grid: numpy array of notes of G grid
    :param psi_grid: numpy array of psi values
    :return: numpy array of phi values that is the upper convex hull function
    of the function psi
    """
    dir_grid = builder.get_directions_grid()
    # print(len(g_grid))
    args = [(m, g_grid, psi_grid, dir_grid)
            for m in range(len(g_grid))]
    pool = Pool(processes=threads)
    data = pool.map(unpacked_min, args)
    pool.close()
    minimals = [el[0] for el in data]
    if not should_track:
        return np.array(minimals), None, None
    indexes = [el[1] for el in data]
    values = [el[2] for el in data]
    return np.array(minimals), indexes, np.array(values)


def unpacked_min(packed_args):
    return get_min(*packed_args)


def get_min(m, g_grid, psi_grid, dir_grid):
    values = [get_body_value(m, d, g_grid, psi_grid, dir_grid)
              for d in range(len(dir_grid))]
    minimal = min(values)
    return minimal, values.index(minimal), values
