"""End to end: synthetic producer -> loopback transport -> consumer."""

import threading
import unittest

import numpy as np

from consumer import Consumer
from models import LinearExtrapolationModel, PersistenceModel
from producer import run_producer
from rasterizer import GridRasterizer
from synthetic_source import SyntheticMITLSource
from tests.common import SMALL_MESH, SMALL_GRID, comm
from transport import LoopbackTransport, StepMessage


def stream(source, consumer, **transport_options):
    transport = LoopbackTransport(**transport_options)
    thread = threading.Thread(target=run_producer, args=(source, transport))
    thread.start()
    records = consumer.run(transport)
    thread.join()
    return records, transport


class PipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = SyntheticMITLSource(comm(), num_steps=30, **SMALL_MESH)
        cls.rast = GridRasterizer(cls.source.mesh, SMALL_GRID, comm(),
                                  "conservative")

    @classmethod
    def tearDownClass(cls):
        del cls.source, cls.rast

    def test_every_step_received_and_conserved(self):
        consumer = Consumer(self.rast, PersistenceModel(), anchor_every=10)
        records, transport = stream(self.source, consumer)
        self.assertEqual([r.step for r in records], list(range(30)))
        for r in records:
            self.assertLess(abs(r.count - r.source_count) / r.source_count,
                            1e-12)
        self.assertEqual(consumer.gaps, 0)
        # Anchors once the window first fills, then every anchor_every steps.
        self.assertEqual(consumer.anchors, [2, 12, 22])

    def test_forecast_lead_and_errors(self):
        consumer = Consumer(self.rast, LinearExtrapolationModel(),
                            anchor_every=10)
        records, _ = stream(self.source, consumer)
        with_forecast = [r for r in records if r.forecast_lead > 0]
        self.assertEqual(len(with_forecast), 30 - 3)
        self.assertEqual(max(r.forecast_lead for r in with_forecast), 10)
        errs = np.array([r.density_error for r in with_forecast])
        self.assertTrue(np.all(np.isfinite(errs)))

    def test_gap_resets_window(self):
        consumer = Consumer(self.rast, PersistenceModel(), anchor_every=10)
        for step in (0, 1, 2, 5, 6, 7, 8):
            consumer.process(StepMessage(step, self.source.time(step),
                                         self.source.fields(step)))
        self.assertEqual(consumer.gaps, 1)
        # Re-anchored on the first full window after the gap (5, 6, 7).
        self.assertEqual(consumer.anchors, [2, 7])

    def test_discard_policy_still_conserves(self):
        consumer = Consumer(self.rast, PersistenceModel(), anchor_every=5)
        records, transport = stream(self.source, consumer, queue_limit=1,
                                    policy="discard")
        self.assertEqual(len(records) + transport.dropped, 30)
        for r in records:
            self.assertLess(abs(r.count - r.source_count) / r.source_count,
                            1e-12)


if __name__ == "__main__":
    unittest.main()
