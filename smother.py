from utils import get_neighbours
import numpy as np


def smooth(values, dim, ann):
    new_values = np.ndarray([len(values)])
    for i in range(len(values)):
        neighbours = list(get_neighbours(i, dim, ann))
        avg = sum([values[n] for n in neighbours]) / len(neighbours)
        new_values[i] = avg
    return new_values
