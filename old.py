import numpy as np


# Test 1
import matplotlib.pyplot as plt

G_grid = get_circle_grid(dim=1, radius=1, grid_ann=100)
psi_grid = np.array([np.sin(4 * G_grid[i]) for i in range(len(G_grid))])
phi_grid = get_upper_convex_hull(G_grid, psi_grid, dir_grid_ann=100)

plt.plot(G_grid, psi_grid)
plt.plot(G_grid, phi_grid)
plt.show()


# Test 2
from mpl_toolkits.mplot3d import Axes3D

def plot_3D(G_grid, f_grid, G_grid_ann):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(G_grid[:, 0].reshape(G_grid_ann, G_grid_ann),
                G_grid[:, 1].reshape(G_grid_ann, G_grid_ann),
                f_grid.reshape(G_grid_ann, G_grid_ann))

G_grid_ann = 10
G_grid = get_circle_grid(dim=2, radius=1, grid_ann=G_grid_ann)
psi_grid = np.array([np.sin(6 * G_grid[i][0]) * np.cos(4 * G_grid[i][1])
                    for i in range(len(G_grid))])
phi_grid = get_upper_convex_hull(G_grid, psi_grid, dir_grid_ann=10)

plot_3D(G_grid, psi_grid, G_grid_ann)
plot_3D(G_grid, phi_grid, G_grid_ann)
plt.show()
