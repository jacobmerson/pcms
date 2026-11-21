"""Needs pcms_testcases exodus/flow/flow.exo; set PCMS_TEST_DATA_DIR to the
pcms_testcases checkout (with the file pulled from Git LFS)."""

import os
import unittest

import numpy as np

from geometry import EMPIRE_STEP_Y, Grid, HEIGHT, LENGTH, STEP_X
from rasterizer import GridRasterizer, make_frame
from tests.common import comm

FLOW = os.path.join(os.environ.get("PCMS_TEST_DATA_DIR", ""), "exodus",
                    "flow", "flow.exo")


def available():
    # An LFS pointer that was never pulled is a ~130 byte text file.
    return os.path.isfile(FLOW) and os.path.getsize(FLOW) > 1_000_000


@unittest.skipUnless(available(), f"{FLOW} not available")
class ExodusSourceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from exodus_source import ExodusRun
        cls.exo = ExodusRun(FLOW, comm())

    @classmethod
    def tearDownClass(cls):
        del cls.exo

    def test_mesh_is_the_mitl_l_shape_in_mm(self):
        self.assertEqual(self.exo.mesh.nelems(), 55065)
        self.assertEqual(self.exo.num_steps, 51)
        area = np.asarray(self.exo.mesh.ask_sizes()).sum()
        self.assertAlmostEqual(area, LENGTH * HEIGHT - STEP_X * EMPIRE_STEP_Y,
                               places=9)

    def test_fields_and_decomposition(self):
        fields = self.exo.fields(50)
        self.assertEqual(set(fields), set(self.exo.field_centering))
        for values in fields.values():
            self.assertEqual(values.shape, (55065,))
        self.assertGreater(fields["n_e"].max(), 0.0)
        np.testing.assert_array_equal(np.unique(self.exo.proc_id()),
                                      np.arange(8))

    def test_conservative_frame_count_is_exact(self):
        # The step at y = 5 mm is on a grid line for ny a multiple of 33.
        grid = Grid(nx=120, ny=33, step_y=EMPIRE_STEP_Y)
        rast = GridRasterizer(self.exo.mesh, grid, comm(), "conservative")
        fields = self.exo.fields(50)
        frame = make_frame(rast, fields, self.exo.field_centering)
        self.assertEqual(frame.shape, (3, 34, 121))
        dens = frame[2]
        exact = rast.source_integral(fields["n_e"])
        self.assertLess(abs(rast.integral(dens) - exact) / exact, 1e-12)
        self.assertGreaterEqual(dens.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
