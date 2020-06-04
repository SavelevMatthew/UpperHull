import matplotlib.pyplot as plt
from grids import *
from upperHull import get_upper_convex_hull

def main():
    builder = GridBuilder(dim=1, ann=100)
    G_grid = builder.get_circle_grid(1)

    psi_grid = np.array([np.sin(4 * G_grid[i]) for i in range(len(G_grid))])

    phi_grid = get_upper_convex_hull(G_grid, psi_grid)

    plt.plot(G_grid, psi_grid)
    plt.plot(G_grid, phi_grid)
    plt.show()


if __name__ == '__main__':
    main()