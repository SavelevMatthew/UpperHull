import matplotlib.pyplot as plt
from grids import *
from upperHull import get_upper_convex_hull

def main():
    G_grid = get_circle_grid(dim=1, radius=1, grid_ann=100)
    psi_grid = np.array([np.sin(4 * G_grid[i]) for i in range(len(G_grid))])

    phi_grid = get_upper_convex_hull(G_grid, psi_grid, dir_grid_ann=100)

    plt.plot(G_grid, psi_grid)
    plt.plot(G_grid, phi_grid)
    plt.show()


if __name__ == '__main__':
    main()