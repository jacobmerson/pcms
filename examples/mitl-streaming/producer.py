"""Producer: replay a run (synthetic now, Exodus later) step by step."""

import time

from transport import StepMessage


def run_producer(source, transport, rate=None, num_steps=None):
    """Send every step of `source` through `transport`, then close it.

    rate: steps per second to emulate simulation pacing (None = as fast as
    possible).
    """
    steps = source.num_steps if num_steps is None else num_steps
    for step in range(steps):
        transport.send(StepMessage(step, source.time(step),
                                   source.fields(step)))
        if rate:
            time.sleep(1.0 / rate)
    transport.close()
