import threading
import unittest

from transport import LoopbackTransport, StepMessage


class LoopbackTransportTest(unittest.TestCase):

    def test_block_delivers_everything_in_order(self):
        t = LoopbackTransport(queue_limit=2, policy="block")

        def produce():
            for step in range(50):
                t.send(StepMessage(step, 0.01 * step))
            t.close()

        thread = threading.Thread(target=produce)
        thread.start()
        steps = [m.step for m in t.messages()]
        thread.join()
        self.assertEqual(steps, list(range(50)))
        self.assertEqual(t.dropped, 0)

    def test_discard_drops_new_steps_when_full(self):
        t = LoopbackTransport(queue_limit=3, policy="discard")
        for step in range(10):          # no consumer yet: queue fills
            t.send(StepMessage(step, 0.0))
        t.close()
        steps = [m.step for m in t.messages()]
        self.assertEqual(steps, [0, 1, 2])
        self.assertEqual((t.sent, t.dropped), (3, 7))


if __name__ == "__main__":
    unittest.main()
