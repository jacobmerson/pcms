import importlib.util
import unittest

import numpy as np

from models import LinearExtrapolationModel, PersistenceModel, PooledTorchModel


def linear_in_time(steps, shape=(2, 8, 12)):
    base = np.random.default_rng(0).normal(size=shape)
    rate = np.random.default_rng(1).normal(size=shape)
    return np.stack([base + s * rate for s in steps]), base, rate


class ModelTest(unittest.TestCase):

    def test_persistence_repeats_newest(self):
        frames, _, _ = linear_in_time([0, 1, 2])
        out = PersistenceModel().rollout(frames, 5)
        self.assertEqual(out.shape, (5,) + frames.shape[1:])
        np.testing.assert_array_equal(out[3], frames[2])

    def test_linear_exact_on_linear_data(self):
        frames, base, rate = linear_in_time([0, 1, 2])
        out = LinearExtrapolationModel().rollout(frames, 4)
        for j in range(4):
            np.testing.assert_allclose(out[j], base + (3 + j) * rate,
                                       atol=1e-12)

    @unittest.skipIf(importlib.util.find_spec("torch") is None,
                     "torch not installed")
    def test_pooled_torch_shape_and_constant_data(self):
        frames = np.full((3, 2, 9, 13), 4.0)
        out = PooledTorchModel(pool=4, device="cpu").rollout(frames, 6)
        self.assertEqual(out.shape, (6, 2, 9, 13))
        np.testing.assert_allclose(out, 4.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
