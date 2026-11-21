#!/usr/bin/env python3
"""Consumer job: receive steps over the PCMS coupler, rasterize them and run
a model in the loop (one rank).

    python run_consumer.py --source exodus --exo flow.exo --plot demo.png
"""

import argparse
import time

import numpy as np

import runtime
from consumer import Consumer
from models import MODELS
from rasterizer import GridRasterizer
from report import plot, summarize
from run_producer import add_transport_args, transport_params
from sources import add_source_args, default_grid, make_source
from transport import PcmsTransport


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_source_args(p)
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--mode", choices=["conservative", "point"],
                   default="conservative")
    p.add_argument("--model", choices=sorted(MODELS) + ["hnode"],
                   default="linear",
                   help="hnode: the pretrained checkpoint (needs 6-channel "
                        "101 x 601 frames, i.e. the synthetic source on the "
                        "default grid; its forecasts are only meaningful on "
                        "EMPIRE data like its training set)")
    p.add_argument("--hnode-dir", default=None,
                   help="hnode checkout with the checkpoint, for --model "
                        "hnode")
    p.add_argument("--anchor-every", type=int, default=10)
    p.add_argument("--plot", default=None, help="write a summary PNG here")
    p.add_argument("--verify", action="store_true",
                   help="check every received field bit for bit against "
                        "the source read locally")
    p.add_argument("--record", default=None,
                   help="write the per-step records (and kept frames) to "
                        "this .npz")
    add_transport_args(p)
    return p.parse_args()


def run(args):
    world = runtime.library().world()
    if world.size() != 1:
        raise SystemExit("the consumer runs on one rank")
    source = make_source(args, world)
    grid = default_grid(args.source, args.nx, args.ny)
    t0 = time.perf_counter()
    rast = GridRasterizer(source.mesh, grid, world, args.mode)
    setup = time.perf_counter() - t0
    n = source.num_steps
    shown = sorted({n // 4, n // 2, n - 1})
    if args.model == "hnode":
        from hnode_model import HNODEModel
        model = HNODEModel(args.hnode_dir)
    else:
        model = MODELS[args.model]()
    consumer = Consumer(rast, model,
                        anchor_every=args.anchor_every, keep_frames=shown,
                        centering=source.field_centering)
    transport = PcmsTransport("consumer", source.mesh,
                              source.field_centering, path=args.path,
                              transport=args.transport,
                              params=transport_params(args))
    print(f"consumer: mesh {source.mesh.nelems()} triangles; grid "
          f"{grid.nx}x{grid.ny} ({args.mode}); rasterizer setup "
          f"{setup:.2f} s; waiting for producer", flush=True)
    t0 = time.perf_counter()
    mismatched = []
    for message in transport.messages():
        if args.verify:
            truth = source.fields(message.step)
            if not all(np.array_equal(message.fields[name], truth[name])
                       for name in truth):
                mismatched.append(message.step)
        consumer.process(message)
    records = consumer.records
    elapsed = time.perf_counter() - t0
    if args.verify:
        print(f"verify: {len(records) - len(mismatched)} of {len(records)} "
              f"steps match the source exactly"
              + (f"; mismatched {mismatched}" if mismatched else ""))
        if mismatched:
            raise SystemExit(1)
    summarize(consumer, records, n, n - len(records), elapsed, args.model)
    if args.plot:
        plot(args.plot, consumer, records)
    if args.record:
        np.savez(args.record,
                 steps=[r.step for r in records],
                 times=[r.time for r in records],
                 count=[r.count for r in records],
                 source_count=[r.source_count for r in records],
                 forecast_count=[r.forecast_count for r in records],
                 density_error=[r.density_error for r in records],
                 mask=rast.mask, weights=rast.weights,
                 **{f"frame_{s}": f for s, (f, _) in consumer.frames.items()})
        print(f"wrote {args.record}")


def main():
    run(parse_args())
    runtime.shutdown()


if __name__ == "__main__":
    main()
