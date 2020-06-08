import matplotlib.pyplot as plt
from grids import *
from upperHull import GD, NP
from multiprocessing import cpu_count
from mpl_toolkits.mplot3d import Axes3D
from timeMeasure import Measurer


def plot_3d(g_grid, f_grid, g_grid_ann):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(g_grid[:, 0].reshape(g_grid_ann, g_grid_ann),
                    g_grid[:, 1].reshape(g_grid_ann, g_grid_ann),
                    f_grid.reshape(g_grid_ann, g_grid_ann))


def main():
    builder = GridBuilder(dim=1, ann=100)

    g_grid = builder.get_circle_grid(1)
    psi_grid = np.array([np.sin(4 * g_grid[i]) for i in range(len(g_grid))])
    phi_grid = GD.get_upper_convex_hull(cpu_count(), builder, g_grid, psi_grid,
                                        10)

    plt.plot(g_grid, psi_grid)
    plt.plot(g_grid, phi_grid)
    plt.show()


def main_3d():
    builder = GridBuilder(2, 10)
    g_grid = builder.get_circle_grid(1)
    psi_grid = np.array([np.sin(6 * g_grid[i][0]) * np.cos(4 * g_grid[i][1])
                         for i in range(len(g_grid))])

    _ = NP.get_upper_convex_hull(cpu_count(), builder, g_grid, psi_grid)
    phi_grid = GD.get_upper_convex_hull(cpu_count(), builder, g_grid, psi_grid,
                                        10)

    plot_3d(g_grid, psi_grid, builder.ann)
    plot_3d(g_grid, phi_grid, builder.ann)
    plt.show()
    print(Measurer.measures)


if __name__ == '__main__':
    main_3d()
