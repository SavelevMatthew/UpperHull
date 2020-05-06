import numpy as np


def get_cubic_grid(dim, min_value, max_value, grid_ann):
    """
    Creates the uniform cubic grid
    :param dim: dimension of the cube
    :param grid_ann: axis notes number - notes number of the grid on each axis
    :param min_value: minimum value for each axis
    :param max_value: maximum value for each axis
    :return: numpy array of notes of the uniform cubic grid
    """
    diameter = (max_value - min_value) / grid_ann
    grid = np.zeros((grid_ann ** dim, dim))

    for i in range(grid_ann ** dim):
        _i = i
        for j in range(dim):
            grid[i][j] = min_value + diameter * (_i % grid_ann + 0.5)
            _i //= grid_ann

    return grid


def get_circle_grid(dim, radius, grid_ann):
    """
    Creates the uniform circle grid
    :param dim: dimension of the circle
    :param radius: radius of the circle
    :param grid_ann: axis notes number
    :return: numpy array of notes of the uniform circle grid
    """
    cubic_grid = get_cubic_grid(dim, -radius, radius, grid_ann)
    #return np.array([value for value in cubic_grid if np.linalg.norm(value) <= radius])
    return cubic_grid

def get_directions_grid(dim, grid_ann):
    """
    Creates auxiliary directional vectors grid
    :param dim: dimension of the vector space
    :param grid_ann: axis notes number
    :return: numpy array of auxiliary directional vectors grid
    """
    angles_grid = get_cubic_grid(dim, 0, np.pi, grid_ann)
    dir_grid = np.zeros(angles_grid.shape)

    for i in range(angles_grid.shape[0]):
        sins = 1
        for j in reversed(range(angles_grid.shape[1])):
            sins *= np.sin(angles_grid[i][j])
            dir_grid[i][j] = np.cos(angles_grid[i][j]) / sins

    return dir_grid