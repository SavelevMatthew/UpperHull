import matplotlib.pyplot as plt
import shutil
from mpl_toolkits.mplot3d import Axes3D
from grids import *
from upperHull import GD, NP
from functions import ThreeDimensions, FourDimensions
from utils import *


def plot_3d(g_grid, f_grid, g_grid_ann):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(g_grid[:, 0].reshape(g_grid_ann, g_grid_ann),
                    g_grid[:, 1].reshape(g_grid_ann, g_grid_ann),
                    f_grid.reshape(g_grid_ann, g_grid_ann))


def process_3d(reports):
    builder = GridBuilder(2, 10, 10)
    g_grid = builder.get_circle_grid(1, 10)
    counter = len(reports) + 1
    path = os.path.join(os.getcwd(), 'report', 'last_graphs')
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    for alpha, func in ThreeDimensions.all:
        func_path = os.path.join(path, 'dim{}_{}'.format(builder.dim + 1,
                                                         func.__name__))
        os.makedirs(func_path)
        psi_grid = np.array([func(*g) for g in g_grid])
        phi_grid_np = NP.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid)
        phi_grid_gd = GD.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid, alpha)
        report = make_report(func.__doc__, counter, builder, phi_grid_np,
                             phi_grid_gd)
        report.insert(4, alpha)
        reports.append(report)
        plot_3d(g_grid, psi_grid, builder.ann)
        plt.savefig(os.path.join(func_path, 'Function.png'))
        plot_3d(g_grid, phi_grid_np, builder.ann)
        plt.savefig(os.path.join(func_path, 'UpperHull_NP.png'))
        plot_3d(g_grid, phi_grid_gd, builder.ann)
        plt.savefig(os.path.join(func_path, 'UpperHull_GD.png'))
        counter += 1
    return reports


def process_4d(reports):
    builder = GridBuilder(3, 5, 5)
    g_grid = builder.get_circle_grid(1, 5)
    counter = len(reports) + 1
    for alpha, func in FourDimensions.all:
        psi_grid = np.array([func(*g) for g in g_grid])
        phi_grid_np = NP.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid)
        phi_grid_gd = GD.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid, alpha)
        report = make_report(func.__doc__, counter, builder, phi_grid_np,
                             phi_grid_gd)
        report.insert(4, alpha)
        reports.append(report)
        counter += 1
    return reports


def main():
    reports = process_3d([])
    reports = process_4d(reports)
    write_statistics(reports)
    plt.show()


if __name__ == '__main__':
    main()
