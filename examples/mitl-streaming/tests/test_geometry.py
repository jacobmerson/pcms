import unittest

import numpy as np

from geometry import (Grid, STEP_X, STEP_Y, compact, inside, mesh_from_arrays,
                      source_mesh, structured_triangles)
from tests.common import SMALL_MESH, SMALL_GRID, comm


class GeometryTest(unittest.TestCase):

    def test_inside_closed_l_shape(self):
        pts = np.array([[0.0, 8.25], [STEP_X, 0.0], [STEP_X, STEP_Y],
                        [10.0, 1.0], [50.0, 0.0], [51.0, 4.0]])
        np.testing.assert_array_equal(
            inside(pts), [True, True, True, False, True, False])

    def test_grid_alignment(self):
        SMALL_GRID.check_step_aligned()
        Grid().check_step_aligned()
        with self.assertRaises(ValueError):
            Grid(nx=120, ny=21).check_step_aligned()

    def test_mesh_round_trip_preserves_order(self):
        x, y = np.linspace(0, 1, 4), np.linspace(0, 2, 3)
        pts, tris = structured_triangles(x, y)
        pts, tris, _ = compact(pts, tris, np.arange(len(tris)) % 3 != 0)
        mesh = mesh_from_arrays(pts, tris, comm())
        np.testing.assert_allclose(mesh.coords().reshape(-1, 2), pts)
        np.testing.assert_array_equal(mesh.ask_elem_verts().reshape(-1, 3),
                                      tris)
        self.assertTrue(mesh.has_tag(0, "global"))

    def test_source_mesh_covers_domain(self):
        mesh = source_mesh(comm(), **SMALL_MESH)
        area = float(np.sum(mesh.ask_sizes()))
        exact = 50.0 * 8.25 - STEP_X * STEP_Y
        self.assertAlmostEqual(area, exact, places=10)
        self.assertGreater(float(np.min(mesh.ask_sizes())), 0.0)


if __name__ == "__main__":
    unittest.main()
