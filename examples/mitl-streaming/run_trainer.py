#!/usr/bin/env python3
"""Training consumer: receive steps over the PCMS coupler and train a
next-frame model per rasterization mode online (one rank).

    python run_trainer.py --source exodus --exo flow.exo --record train.npz
"""

import argparse
import os
import time

import numpy as np

import runtime
from online_model import Featurizer, OnlineRecord, OnlineTrainer
from rasterizer import GridRasterizer, frame_channels, make_frame
from run_producer import add_transport_args, transport_params
from sources import add_source_args, default_grid, make_source
from transport import PcmsTransport

MODES = ("point", "conservative")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_source_args(p)
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--model", choices=["cnn", "hnode"], default="cnn")
    p.add_argument("--hnode-dir", default=None,
                   help="hnode checkout with the checkpoint, for --model "
                        "hnode")
    p.add_argument("--hnode-steps", type=int, default=None,
                   help="stop fine-tuning hnode after this many steps")
    p.add_argument("--train-steps", type=int, default=20,
                   help="optimizer steps per received frame")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default: 1e-3 cnn, 1e-6 hnode)")
    p.add_argument("--horizon", type=int, default=5,
                   help="lead of the multi-step forecast scored on arrival")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=None,
                   help="torch threads (default: the cores this process may "
                        "use; under mpiexec torch would otherwise use one)")
    p.add_argument("--record", default=None,
                   help="write the per-step record to this .npz "
                        "(input of make_video.py)")
    add_transport_args(p)
    return p.parse_args()


def run(args):
    import torch
    torch.set_num_threads(args.threads or len(os.sched_getaffinity(0)))
    world = runtime.library().world()
    if world.size() != 1:
        raise SystemExit("the trainer runs on one rank")
    source = make_source(args, world)
    if args.model == "hnode":
        channels = frame_channels(source.field_centering)
        if len(channels) != 6:
            raise SystemExit(
                f"--model hnode needs the 6 channels Ex, Ey, Bz, vx, vy, n_e; "
                f"--source {args.source} provides {', '.join(channels)}")
    grid = default_grid(args.source, args.nx, args.ny)
    rasts = {m: GridRasterizer(source.mesh, grid, world, m)
             for m in args.modes}
    transport = PcmsTransport("consumer", source.mesh,
                              source.field_centering, path=args.path,
                              transport=args.transport,
                              params=transport_params(args))
    if args.model == "hnode":
        return run_hnode(args, rasts[args.modes[0]], source, transport)
    trainers = {}
    feat = None               # one Featurizer shared by every mode
    record = OnlineRecord()
    meta = {"step": [], "time": [], "source_count": [], "ingest_time": []}
    print(f"trainer: mesh {source.mesh.nelems()} triangles; grid "
          f"{grid.nx}x{grid.ny}; modes {', '.join(args.modes)}; "
          f"{args.train_steps} optimizer steps per frame; waiting for "
          f"producer", flush=True)
    t_start = time.perf_counter()
    for message in transport.messages():
        meta["step"].append(message.step)
        meta["time"].append(message.time)
        meta["source_count"].append(
            next(iter(rasts.values())).source_integral(message.fields["n_e"]))
        line = [f"step {message.step:3d}"]
        frames = {}
        t0 = time.perf_counter()
        for mode, rast in rasts.items():
            frames[mode] = make_frame(rast, message.fields,
                                      source.field_centering)
        ingest = time.perf_counter() - t0
        channels = frame_channels(message.fields)
        if feat is None:
            feat = Featurizer(channels, channels.index("n_e"))
        # all modes update the shared scale before any model scores
        for mode, frame in frames.items():
            feat.observe(frame, rasts[mode].mask)
        for mode, rast in rasts.items():
            frame = frames[mode]
            if mode not in trainers:
                trainers[mode] = OnlineTrainer(
                    channels, channels.index("n_e"), rast.mask,
                    steps_per_frame=args.train_steps,
                    batch_size=args.batch_size, lr=args.lr or 1e-3,
                    horizon=args.horizon, seed=args.seed, featurizer=feat)
            result = trainers[mode].receive(frame)
            record.add(mode, trainers[mode], result)
            record.rows[f"{mode}_count"].append(
                rast.integral(frame[channels.index("n_e")]))
            line.append(f"{mode}: loss {result['loss']:.4f} err1 "
                        f"{result['err1']:.4f} (persist "
                        f"{result['err1_persist']:.4f})")
        meta["ingest_time"].append(ingest)
        print("  ".join(line), flush=True)
    elapsed = time.perf_counter() - t_start
    n = len(meta["step"])
    print(f"received {n} steps in {elapsed:.1f} s "
          f"({elapsed / max(n, 1):.2f} s/step incl. training)")
    for mode, tr in trainers.items():
        rows = record.rows
        e1 = np.array(rows[f"{mode}_err1"])
        p1 = np.array(rows[f"{mode}_err1_persist"])
        late = slice(n // 2, None)
        print(f"{mode}: {tr.optimizer_steps} optimizer steps; one-step "
              f"error / persistence, second half of the run: "
              f"{np.nanmean(e1[late]) / np.nanmean(p1[late]):.3f}")
    if args.record:
        np.savez_compressed(
            args.record, modes=np.array(args.modes), horizon=args.horizon,
            train_steps=args.train_steps,
            extent=[grid.x0, grid.x1, grid.y0, grid.y1],
            **{f"{m}_mask": r.mask for m, r in rasts.items()},
            **{f"{m}_losses": np.array(t.losses)
               for m, t in trainers.items()},
            **{k: np.asarray(v) for k, v in meta.items()},
            **record.arrays())
        print(f"wrote {args.record}")


def run_hnode(args, rast, source, transport):
    from hnode_model import HNODEModel, HNODEStreamTrainer
    from rasterizer import DENSITY_CHANNEL
    model = HNODEModel(args.hnode_dir)
    lr = args.lr or 1e-6
    trainer = HNODEStreamTrainer(model, lr=lr, max_steps=args.hnode_steps)
    rows = {k: [] for k in ("step", "time", "loss", "train_time", "truth",
                            "pred", "persist")}

    def log_density(f):
        return np.log10(np.maximum(f[DENSITY_CHANNEL], 0.0)
                        + 1e12).astype(np.float32)

    print(f"trainer: hnode fine-tuning on {args.modes[0]} frames, lr {lr}; "
          f"waiting for producer", flush=True)
    for message in transport.messages():
        frame = make_frame(rast, message.fields, source.field_centering)
        result = trainer.receive(frame)
        rows["step"].append(message.step)
        rows["time"].append(message.time)
        rows["truth"].append(log_density(frame))
        if result is None:
            print(f"step {message.step:3d}: collecting frames", flush=True)
            for k in ("loss", "train_time"):
                rows[k].append(np.nan)
            rows["pred"].append(np.full(frame.shape[1:], np.nan, np.float32))
            rows["persist"].append(np.full(frame.shape[1:], np.nan,
                                           np.float32))
            continue
        loss, seconds, pred = result
        rows["loss"].append(loss)
        rows["train_time"].append(seconds)
        rows["pred"].append(log_density(pred))
        rows["persist"].append(log_density(
            trainer.frames[-1 - max(trainer.leads)]))
        print(f"step {message.step:3d}: hnode optimizer step "
              f"{len(trainer.losses)}, loss {loss:.4e}, {seconds:.1f} s",
              flush=True)
    if args.record:
        np.savez_compressed(
            args.record, model="hnode", mode=args.modes[0],
            mask=rast.mask, lead=max(trainer.leads),
            extent=[rast.grid.x0, rast.grid.x1, rast.grid.y0, rast.grid.y1],
            **{k: np.asarray(v) for k, v in rows.items()})
        print(f"wrote {args.record}")


def main():
    run(parse_args())
    runtime.shutdown()


if __name__ == "__main__":
    main()
