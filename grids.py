import numpy as np


class GridBuilder:
    def __init__(self, dim, ann):
        self.dim = dim
        self.ann = ann
        self.divs = [self.ann ** i for i in range(self.dim)]

    def get_directions_grid(self):
        """
        Creates auxiliary directional vectors grid
        :return: numpy array of auxiliary directional vectors grid
        """
        angles_grid = self.get_cubic_grid(0, np.pi)
        dir_grid = np.zeros(angles_grid.shape)

        for i in range(angles_grid.shape[0]):
            sins = 1
            for j in reversed(range(angles_grid.shape[1])):
                sins *= np.sin(angles_grid[i][j])
                dir_grid[i][j] = np.cos(angles_grid[i][j]) / sins

        return dir_grid

    def get_circle_grid(self, radius):
        """
        Creates the uniform circle grid
        :param radius: radius of the circle
        :return: numpy array of notes of the uniform circle grid
        """
        cubic_grid = self.get_cubic_grid(-radius, radius)
        # return np.array([value for value in cubic_grid if np.linalg.norm(value) <= radius])
        return cubic_grid

    def get_cubic_grid(self, min_value, max_value):
        """
        Creates the uniform cubic grid
        :param min_value: minimum value for each axis
        :param max_value: maximum value for each axis
        :return: numpy array of notes of the uniform cubic grid
        """
        diameter = (max_value - min_value) / self.ann
        grid = np.zeros((self.ann ** self.dim, self.dim))

        for i in range(self.ann ** self.dim):
            _i = i
            for j in range(self.dim):
                grid[i][j] = min_value + diameter * (_i % self.ann + 0.5)
                _i //= self.ann

        return grid

    def get_neighbours(self, index):
        result = []
        for div in self.divs:
            m = (index // div) % self.ann
            if m != 0:
                result.append(index - div)
            if m != self.ann - 1:
                result.append(index + div)
        return result
