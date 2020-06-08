import matplotlib.pyplot as plt
from grids import *
from upperHull import GD, NP
from multiprocessing import cpu_count
from nicePrinters import *
from mpl_toolkits.mplot3d import Axes3D
from measure import Measurer


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
    phi_grid_np = NP.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                           psi_grid)

    plt.plot(g_grid, psi_grid)
    plt.plot(g_grid, phi_grid)
    plt.plot(g_grid, phi_grid_np)
    plt.show()


def main_3d():
    builder = GridBuilder(2, 10)
    g_grid = builder.get_circle_grid(1)
    functions_grids = get_3d_functions(g_grid)
    counter = 1
    for psi_grid in functions_grids:
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


def make_report(counter, phi_np, phi_gd):
    print('=' * 64)
    info('Функция #{} (Потоков: {})'.format(counter, cpu_count()))
    info('Время полного перебора: {} секунд'.format(Measurer
                                                    .time_measures[0]
                                                    .total_seconds()))
    info('Время градиентного спуска: {} секунд'.format(Measurer
                                                       .time_measures[1]
                                                       .total_seconds()))
    max_error = round(max([abs(phi_gd[j] - phi_np[j])
                           for j in range(len(phi_gd))]), 5)
    info('Максимальная ошибка: {}'.format(max_error))
    avg_error = round(sum([abs(phi_gd[j] - phi_np[j])
                           for j in range(len(phi_gd))])
                      / len(phi_gd), 5)
    info('Средняя ошибка: {}'.format(avg_error))
    counter += 1
    Measurer.time_measures.clear()
    print('=' * 64)


def get_3d_functions(g_grid):
    functions_grids = []
    coords = [(g_grid[i][0], g_grid[i][1]) for i in range(len(g_grid))]
    functions_grids.append(np.array([np.sin(6 * x) *
                                     np.cos(4 * y)
                                     for x, y in coords]))
    functions_grids.append(np.array([(x ** 2 + y ** 2)
                                     for x, y in coords]))

    functions_grids.append(np.array([(np.sin(y * 2) * y * 5)
                                     for x, y in coords]))
    functions_grids.append(np.array([2 * np.cos(x * 2) * np.cos(y)
                                     for x, y in coords]))
    functions_grids.append(np.array([(0.26 * ((10 * x) ** 2 + (10 * y) ** 2)
                                      - 48 * x * y) / 100
                                     for x, y in coords]))

    return functions_grids


if __name__ == '__main__':
    main_3d()
