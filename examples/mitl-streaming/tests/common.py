"""Small, fast fixtures shared by the tests."""

import runtime
from geometry import Grid

# ~6k-triangle source mesh and a 120 x 20 frame grid (step on grid lines).
SMALL_MESH = dict(nx_left=35, nx_right=25, ny_low=7, ny_high=10)
SMALL_GRID = Grid(nx=120, ny=20)


def comm():
    return runtime.library().world()
