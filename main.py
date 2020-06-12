import matplotlib.pyplot as plt
from grids import *
from upperHull import GD, NP
from functions import ThreeDimensions
from nicePrinters import *


def plot_3d(g_grid, f_grid, g_grid_ann):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(g_grid[:, 0].reshape(g_grid_ann, g_grid_ann),
                    g_grid[:, 1].reshape(g_grid_ann, g_grid_ann),
                    f_grid.reshape(g_grid_ann, g_grid_ann))


def main():
    builder = GridBuilder(2, 10)
    g_grid = builder.get_circle_grid(1)
    counter = 1
    for alpha, func in ThreeDimensions.all:
        psi_grid = np.array([func(*g) for g in g_grid])
        phi_grid_np = NP.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid)
        phi_grid_gd = GD.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid, 10)
        make_report(counter, phi_grid_np, phi_grid_gd)
        plot_3d(g_grid, psi_grid, builder.ann)
        plot_3d(g_grid, phi_grid_np, builder.ann)
        plot_3d(g_grid, phi_grid_gd, builder.ann)
        counter += 1
    plt.show()


if __name__ == '__main__':
    main()
