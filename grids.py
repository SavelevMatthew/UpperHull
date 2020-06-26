import numpy as np


class GridBuilder:
    def __init__(self, dim, ann, dir_ann, accuracy=100):
        self.dim = dim
        self.ann = ann
        self.dir_ann = dir_ann
        self.divs = [self.ann ** i for i in range(self.dim)]
        self.acc = accuracy

    def get_directions_grid(self):
        """
        Creates auxiliary directional vectors grid
        :return: numpy array of auxiliary directional vectors grid
        """
        angles_grid = self.get_cubic_grid(0, np.pi, self.dir_ann)
        dir_grid = np.zeros(angles_grid.shape)

        for i in range(angles_grid.shape[0]):
            sins = 1
            for j in reversed(range(angles_grid.shape[1])):
                sins *= np.sin(angles_grid[i][j])
                dir_grid[i][j] = np.cos(angles_grid[i][j]) / sins

        return dir_grid

    def get_circle_grid(self, radius, ann):
        """
        Creates the uniform circle grid
        :param radius: radius of the circle
        :return: numpy array of notes of the uniform circle grid
        """
        cubic_grid = self.get_cubic_grid(-radius, radius, ann)
        # return np.array([value for value in cubic_grid if np.linalg.norm(value) <= radius])
        return cubic_grid

    def get_cubic_grid(self, min_value, max_value, ann):
        """
        Creates the uniform cubic grid
        :param ann: number of dots in every dimension
        :param min_value: minimum value for each axis
        :param max_value: maximum value for each axis
        :return: numpy array of notes of the uniform cubic grid
        """
        diameter = (max_value - min_value) / ann
        grid = np.zeros((ann ** self.dim, self.dim))

        for i in range(ann ** self.dim):
            _i = i
            for j in range(self.dim):
                grid[i][j] = min_value + diameter * (_i % ann + 0.5)
                _i //= ann

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
