#!/usr/bin/env python3
"""Streaming demo in one process: producer thread, loopback transport,
consumer.
"""

import argparse
import threading
import time

import runtime
from consumer import Consumer
from geometry import Grid
from models import MODELS
from producer import run_producer
from rasterizer import GridRasterizer
from report import plot, summarize
from synthetic_source import SyntheticMITLSource
from transport import LoopbackTransport


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--nx", type=int, default=600)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--mode", choices=["conservative", "point"],
                   default="conservative")
    p.add_argument("--model", choices=sorted(MODELS), default="linear")
    p.add_argument("--anchor-every", type=int, default=20)
    p.add_argument("--rate", type=float, default=None,
                   help="producer steps per second (default: unthrottled)")
    p.add_argument("--queue-limit", type=int, default=4)
    p.add_argument("--policy", choices=["block", "discard"], default="block")
    p.add_argument("--plot", default=None, help="write a summary PNG here")
    return p.parse_args()


def run(args):
    comm = runtime.library().world()
    source = SyntheticMITLSource(comm, num_steps=args.steps)
    grid = Grid(nx=args.nx, ny=args.ny)
    t0 = time.perf_counter()
    rast = GridRasterizer(source.mesh, grid, comm, args.mode)
    setup = time.perf_counter() - t0
    shown = [s for s in (args.steps // 4, args.steps // 2, args.steps - 1)]
    consumer = Consumer(rast, MODELS[args.model](),
                        anchor_every=args.anchor_every, keep_frames=shown)

    transport = LoopbackTransport(args.queue_limit, args.policy)
    producer = threading.Thread(target=run_producer,
                                args=(source, transport, args.rate))
    t0 = time.perf_counter()
    producer.start()
    records = consumer.run(transport)
    producer.join()
    elapsed = time.perf_counter() - t0

    print(f"mesh: {source.mesh.nelems()} triangles; grid {grid.nx}x{grid.ny} "
          f"({args.mode}); rasterizer setup {setup:.2f} s")
    summarize(consumer, records, args.steps, transport.dropped, elapsed,
              args.model)
    if args.plot:
        plot(args.plot, consumer, records)


def main():
    args = parse_args()
    run(args)
    runtime.shutdown()


if __name__ == "__main__":
    main()
