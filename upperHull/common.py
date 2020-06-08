def get_max(d, g_grid, psi_grid, dir_grid):
    return max([get_inner_value(d, g, g_grid, psi_grid, dir_grid)
                for g in range(len(g_grid))])


def get_body_value(m, d, g_grid, psi_grid, dir_grid):
    return (get_max(d, g_grid, psi_grid, dir_grid)
            - sum(dir_grid[d] * g_grid[m]))


def get_inner_value(d, g, g_grid, psi_grid, dir_grid):
    return psi_grid[g] + sum(dir_grid[d] * g_grid[g])
