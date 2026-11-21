"""Producer: send a run step by step."""

import time

from transport import StepMessage


def run_producer(source, transport, rate=None, num_steps=None):
    """Send every step of `source` through `transport`, then close it.

    Args:
        rate: steps per second (None: as fast as possible).
        num_steps: steps to send (default all).
    """
    steps = source.num_steps if num_steps is None else num_steps
    for step in range(steps):
        transport.send(StepMessage(step, source.time(step),
                                   source.fields(step)))
        if rate:
            time.sleep(1.0 / rate)
    transport.close()
