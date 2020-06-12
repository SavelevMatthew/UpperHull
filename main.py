import matplotlib.pyplot as plt
from grids import *
from upperHull import GD, NP
from functions import ThreeDimensions
from utils import *
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(g_grid, f_grid, g_grid_ann):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(g_grid[:, 0].reshape(g_grid_ann, g_grid_ann),
                    g_grid[:, 1].reshape(g_grid_ann, g_grid_ann),
                    f_grid.reshape(g_grid_ann, g_grid_ann))


def main():
    builder = GridBuilder(2, 10)
    g_grid = builder.get_circle_grid(1)
    reports = []
    counter = 1
    for alpha, func in ThreeDimensions.all:
        psi_grid = np.array([func(*g) for g in g_grid])
        phi_grid_np = NP.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid)
        phi_grid_gd = GD.get_upper_convex_hull(cpu_count(), builder, g_grid,
                                               psi_grid, 10)
        report = make_report(func.__name__, counter, builder, phi_grid_np,
                             phi_grid_gd)
        report.insert(4, alpha)
        reports.append(report)
        plot_3d(g_grid, psi_grid, builder.ann)
        plot_3d(g_grid, phi_grid_np, builder.ann)
        plot_3d(g_grid, phi_grid_gd, builder.ann)
        counter += 1
    plt.show()

    df = pd.DataFrame(reports, columns=['Function Name', 'Dimensions',
                                        'Threads', 'Dots amount', 'Alpha',
                                        'Full NP time (sec)', 'GD time (sec)',
                                        'Max error (%)', 'Average Error (%)'])
    path, sheet = get_available_name()

    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name=sheet)
    worksheet = writer.sheets[sheet]
    worksheet.set_zoom(50)
    worksheet.set_column('A:A', 32)
    worksheet.set_column('B:D', 16)
    worksheet.set_column('E:H', 24)
    writer.save()


if __name__ == '__main__':
    main()
