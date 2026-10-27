import unittest

import numpy as np

from rasterizer import (CHANNELS, DENSITY_CHANNEL, LUMPED, GridRasterizer,
                        make_frame)
from synthetic_source import SyntheticMITLSource
from tests.common import SMALL_MESH, SMALL_GRID, comm


class RasterizerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = SyntheticMITLSource(comm(), num_steps=40, **SMALL_MESH)
        cls.cons = GridRasterizer(cls.source.mesh, SMALL_GRID, comm(),
                                  "conservative")
        cls.point = GridRasterizer(cls.source.mesh, SMALL_GRID, comm(),
                                   "point")

    @classmethod
    def tearDownClass(cls):
        del cls.source, cls.cons, cls.point

    def test_constant_maps_to_one_on_mask(self):
        ones = np.ones(self.source.mesh.nelems())
        for rast in (self.cons, self.point):
            frame = rast.map(ones, "elem", LUMPED)
            np.testing.assert_allclose(frame[rast.mask], 1.0, atol=1e-12)
            self.assertTrue(np.all(frame[~rast.mask] == 0.0))

    def test_weights_integrate_domain_area(self):
        area = 50.0 * 8.25 - 35.0 * 3.3
        self.assertAlmostEqual(float(self.cons.weights.sum()), area,
                               places=10)

    def test_conservative_density_is_exact_and_positive(self):
        fields = self.source.fields(30)
        frame = make_frame(self.cons, fields)
        dens = frame[DENSITY_CHANNEL]
        exact = self.cons.source_integral(fields["n_e"])
        self.assertLess(abs(self.cons.integral(dens) - exact) / exact, 1e-12)
        self.assertGreaterEqual(float(dens[self.cons.mask].min()), 0.0)

    def test_point_sampling_is_not_conservative(self):
        fields = self.source.fields(30)
        dens = make_frame(self.point, fields)[DENSITY_CHANNEL]
        exact = self.point.source_integral(fields["n_e"])
        self.assertGreater(abs(self.point.integral(dens) - exact) / exact,
                           1e-4)

    def test_point_mode_reproduces_linear_nodal_field(self):
        coords = self.source.mesh.coords().reshape(-1, 2)
        linear = 2.0 * coords[:, 0] - 3.0 * coords[:, 1] + 1.0
        frame = self.point.map(linear, "node")
        nodes = SMALL_GRID.node_coords().reshape(*SMALL_GRID.shape, 2)
        expect = 2.0 * nodes[..., 0] - 3.0 * nodes[..., 1] + 1.0
        np.testing.assert_allclose(frame[self.point.mask],
                                   expect[self.point.mask], atol=1e-9)

    def test_frame_layout(self):
        frame = make_frame(self.cons, self.source.fields(5))
        self.assertEqual(frame.shape, (len(CHANNELS),) + SMALL_GRID.shape)
        self.assertEqual(frame.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(frame)))


if __name__ == "__main__":
    unittest.main()
