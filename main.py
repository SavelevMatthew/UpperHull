import matplotlib.pyplot as plt
from grids import *
from upperHull import get_upper_convex_hull
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(G_grid, f_grid, G_grid_ann):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(G_grid[:, 0].reshape(G_grid_ann, G_grid_ann),
                    G_grid[:, 1].reshape(G_grid_ann, G_grid_ann),
                    f_grid.reshape(G_grid_ann, G_grid_ann))


def main():
    builder = GridBuilder(dim=1, ann=100)
    G_grid = builder.get_circle_grid(1)

    psi_grid = np.array([np.sin(4 * G_grid[i]) for i in range(len(G_grid))])

    phi_grid = get_upper_convex_hull(builder, G_grid, psi_grid)

    plt.plot(G_grid, psi_grid)
    plt.plot(G_grid, phi_grid)
    plt.show()


def main_3d():
    builder = GridBuilder(2, 10)
    G_grid = builder.get_circle_grid(1)
    psi_grid = np.array([np.sin(6 * G_grid[i][0]) * np.cos(4 * G_grid[i][1])
                         for i in range(len(G_grid))])
    phi_grid = get_upper_convex_hull(builder, G_grid, psi_grid)

    plot_3d(G_grid, psi_grid, builder.ann)
    plot_3d(G_grid, phi_grid, builder.ann)
    plt.show()


if __name__ == '__main__':
    main_3d()