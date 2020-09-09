import matplotlib.pyplot as plt
import shutil
from mpl_toolkits.mplot3d import Axes3D
from grids import *
from upperHull import GD, NP
from functions import ThreeDimensions, FourDimensions, TwoDimensions
from utils import *
from smother import smooth


def plot_3d(g_grid, f_grid, g_grid_ann):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(g_grid[:, 0].reshape(g_grid_ann, g_grid_ann),
                    g_grid[:, 1].reshape(g_grid_ann, g_grid_ann),
                    f_grid.reshape(g_grid_ann, g_grid_ann))


def process_2d(reports):
    builder = GridBuilder(1, 100, 100, 1)
    g_grid = builder.get_circle_grid(100)
    counter = len(reports) + 1
    path = os.path.join(os.getcwd(), 'report', 'last_graphs')
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    for alpha, func in TwoDimensions.all:
        func_path = os.path.join(path, 'dim{}_{}.png'.format(builder.dim + 1,
                                                         func.__name__))
        psi_grid = np.array([func(*g) for g in g_grid])
        phi_grid_np, *_ = NP.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid)
        phi_grid_gd, = GD.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid, alpha)
        report = make_report(func.__doc__, counter, builder, phi_grid_np,
                             phi_grid_gd)
        report.insert(5, alpha)
        reports.append(report)
        f_graph, = plt.plot(g_grid, psi_grid, label='Функция')
        np_graph, = plt.plot(g_grid, phi_grid_np, label='Перебор')
        gd_graph, = plt.plot(g_grid, phi_grid_gd, label='Спуск')
        plt.legend(handles=[f_graph, np_graph, gd_graph])
        plt.savefig(func_path)
        plt.close()
        counter += 1
    return reports


def process_3d(reports):
    builder = GridBuilder(2, 15, 15, 1, 20)
    g_grid = builder.get_circle_grid(15)
    counter = len(reports) + 1
    path = os.path.join(os.getcwd(), 'report', 'last_graphs')
    if os.path.exists(path) and os.path.isdir(path) and len(reports) == 0:
        shutil.rmtree(path, ignore_errors=True)
    for alpha, func in ThreeDimensions.all:
        func_path = os.path.join(path, 'dim{}_{}'.format(builder.dim + 1,
                                                         func.__name__))
        os.makedirs(func_path)
        psi_grid = np.array([func(*g) for g in g_grid])
        should_track = func in ThreeDimensions.tracking
        phi_grid_np, minimals, values = NP.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                                         psi_grid, should_track)
        phi_grid_gd, tracks = GD.get_upper_convex_hull(cpu_count(), builder, g_grid, psi_grid, alpha, should_track)
        # phi_grid_gd = smooth(phi_grid_gd, builder.dim, builder.ann)
        report = make_report(func.__doc__, counter, builder, phi_grid_np,
                             phi_grid_gd)
        report.insert(5, alpha)
        reports.append(report)
        plot_3d(g_grid, psi_grid, builder.ann)
        plt.savefig(os.path.join(func_path, 'Function.png'))
        plot_3d(g_grid, phi_grid_np, builder.ann)
        plt.savefig(os.path.join(func_path, 'UpperHull_NP.png'))
        plot_3d(g_grid, phi_grid_gd, builder.ann)
        plt.savefig(os.path.join(func_path, 'UpperHull_GD.png'))
        if should_track:
            d_grid = builder.get_directions_grid()
            for i in range(len(g_grid)):
                # Отрисовка внутренней функции
                plt.figure()
                plot_3d(d_grid, values[i], builder.dir_ann)
                plt.figure()
                # конец отрисовки
                path = np.array([d_grid[j] for j in tracks[i]])
                plt.plot(path[:, 0], path[:, 1])
                min_pos = d_grid[minimals[i]]
                last_post = path[len(path) - 1]
                path_finish = np.array([last_post, last_post])
                actual_finish = np.array([min_pos, min_pos])
                plt.plot(path_finish[:, 0], path_finish[:, 1], 'bo')
                plt.plot(actual_finish[:, 0], actual_finish[:, 1], 'ro')
                plt.show()
        counter += 1
    return reports


def process_4d(reports):
    builder = GridBuilder(3, 5, 5, 1)
    g_grid = builder.get_circle_grid(5)
    counter = len(reports) + 1
    for alpha, func in FourDimensions.all:
        psi_grid = np.array([func(*g) for g in g_grid])
        phi_grid_np, *_ = NP.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid)
        phi_grid_gd = GD.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid, alpha)
        report = make_report(func.__doc__, counter, builder, phi_grid_np,
                             phi_grid_gd)
        report.insert(5, alpha)
        reports.append(report)
        counter += 1
    return reports


def main():
    reports = []
    #reports = process_2d(reports)
    reports = process_3d(reports)
    #reports = process_4d(reports)
    write_statistics(reports)
    plt.show()


if __name__ == '__main__':
    main()
