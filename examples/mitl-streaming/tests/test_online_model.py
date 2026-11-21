import importlib.util
import unittest

import numpy as np


@unittest.skipIf(importlib.util.find_spec("torch") is None,
                 "torch not installed")
class OnlineTrainerTest(unittest.TestCase):

    def frames(self, n, shape=(3, 16, 40)):
        # A density blob and a velocity field moving one pixel per step.
        y, x = np.mgrid[:shape[1], :shape[2]]
        out = []
        for t in range(n):
            blob = np.exp(-((x - 8 - t) ** 2 + (y - 8) ** 2) / 8.0)
            out.append(np.stack([np.cos(x / 5.0 - t), np.sin(y / 3.0),
                                 1e16 * blob + 1e13]))
        return out

    def test_untrained_is_persistence_and_training_runs(self):
        from online_model import OnlineTrainer
        mask = np.ones((16, 40), dtype=bool)
        mask[:4, :10] = False
        tr = OnlineTrainer(("vx", "vy", "n_e"), 2, mask, steps_per_frame=0,
                           crop=(8, 16), horizon=2)
        results = [tr.receive(f) for f in self.frames(3)]
        # zero output layer: persistence, nothing outside the mask
        self.assertAlmostEqual(results[2]["err1"],
                               results[2]["err1_persist"], places=6)
        self.assertTrue(np.all(results[2]["pred1"][:, ~mask] == 0))
        tr.steps_per_frame = 3
        r = tr.receive(self.frames(4)[3])
        self.assertEqual(tr.optimizer_steps, 3)
        self.assertTrue(np.isfinite(r["loss"]))
        self.assertIsNotNone(r["predh"])        # forecast made at step 1

    def test_crop_larger_than_frame(self):
        from online_model import OnlineTrainer
        mask = np.ones((16, 40), dtype=bool)
        tr = OnlineTrainer(("vx", "vy", "n_e"), 2, mask, steps_per_frame=2,
                           crop=(64, 192))
        for f in self.frames(4):
            r = tr.receive(f)
        self.assertEqual(tr.optimizer_steps, 4)
        self.assertTrue(np.isfinite(r["loss"]))

    def test_shared_featurizer_makes_models_identical(self):
        # a shared featurizer makes the result independent of trainer order
        from online_model import Featurizer, OnlineTrainer
        mask = np.ones((16, 40), dtype=bool)
        feat = Featurizer(("vx", "vy", "n_e"), 2)
        trainers = [OnlineTrainer(("vx", "vy", "n_e"), 2, mask,
                                  steps_per_frame=2, crop=(8, 16), horizon=2,
                                  featurizer=feat) for _ in range(2)]
        for f in self.frames(6):
            feat.observe(f, mask)
            a, b = (tr.receive(f) for tr in trainers)
            for key in ("err1", "errh", "loss"):
                np.testing.assert_equal(a[key], b[key])

    def test_forecast_scored_in_current_scale(self):
        # scale grows after the forecast: must still score as persistence
        from online_model import OnlineTrainer
        mask = np.ones((16, 40), dtype=bool)
        tr = OnlineTrainer(("vx", "vy", "n_e"), 2, mask, steps_per_frame=0,
                           horizon=2)
        frames = self.frames(4)
        frames[3] = frames[3].copy()
        frames[3][0] *= 4.0
        for f in frames:
            r = tr.receive(f)
        self.assertAlmostEqual(r["errh"], r["errh_persist"], places=6)

    def test_density_features_round_trip(self):
        from online_model import Featurizer
        f = Featurizer(("vx", "vy", "n_e"), 2)
        frame = self.frames(1)[0]
        mask = np.ones(frame.shape[1:], dtype=bool)
        f.observe(frame, mask)
        feats = f(frame)
        np.testing.assert_allclose(
            f.log_density(feats), np.log10(frame[2] + 1e12), rtol=1e-6)
        self.assertLessEqual(np.abs(feats[0]).max(), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
