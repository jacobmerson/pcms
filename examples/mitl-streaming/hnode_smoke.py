#!/usr/bin/env python3
"""Smoke test of the hnode adapter on the frames bundled with hnode.

Forecasts frames 3..9 of plasma_gun_frames/example_traj from frames 0..2
through the same rollout() call the consumer uses, and compares them with
the stored frames, and with persistence, in the training metric. Runs on the
CPU (slowly) when no GPU is present.

    python hnode_smoke.py /path/to/hnode
"""

import argparse
import os
import time

import numpy as np
import torch

from hnode_model import HNODEModel
from models import PersistenceModel
from rasterizer import CHANNELS


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("hnode_dir")
    p.add_argument("--plot", default=None)
    args = p.parse_args()

    frames_dir = os.path.join(args.hnode_dir, "plasma_gun_frames",
                              "example_traj", "frames")
    frames = np.stack([
        torch.load(os.path.join(frames_dir, f"frame_{i:06d}.pt")).double()
        .numpy() for i in range(10)])
    t0 = time.perf_counter()
    model = HNODEModel(args.hnode_dir)
    print(f"loaded checkpoint in {time.perf_counter() - t0:.1f} s on "
          f"{model.device}")
    mask = model.mask.numpy() > 0

    horizon = len(frames) - 3
    t0 = time.perf_counter()
    forecast = model.rollout(frames[:3], horizon)
    print(f"rollout of {horizon} frames in {time.perf_counter() - t0:.1f} s")
    baseline = PersistenceModel().rollout(frames[:3], horizon)
    assert forecast.shape == frames[3:].shape

    # The training metric: masked L1 of the normalized channels. Relative
    # errors are meaningless here: the bundled frames are early, nearly
    # static and close to zero in most channels.
    std = model.std.numpy()

    def l1(pred, truth):
        return np.abs(((pred - truth) / std)[..., mask]).mean(axis=(-1, -2))

    print("normalized masked L1 per lead (hnode / persistence):")
    for j, (e_h, e_p) in enumerate(zip(l1(forecast, frames[3:]),
                                       l1(baseline, frames[3:]))):
        print(f"  lead {j + 1}: {e_h:.2e} / {e_p:.2e}")
    print("for scale: the checkpoint's training loss (same metric) ends "
          "near 7e-2")
    if not np.all(np.isfinite(forecast)):
        raise SystemExit("non-finite forecast")
    if args.plot:
        plot(args.plot, frames, forecast, mask)


def plot(path, frames, forecast, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rho = CHANNELS.index("n_e")
    leads = [0, len(forecast) // 2, len(forecast) - 1]
    fig, axes = plt.subplots(len(leads), 2, figsize=(11, 1.5 * len(leads)))
    for row, j in enumerate(leads):
        for col, (img, label) in enumerate(((frames[3 + j], "stored"),
                                            (forecast[j], "hnode"))):
            ax = axes[row, col]
            ax.imshow(np.where(mask, img[rho], np.nan), origin="lower",
                      cmap="inferno", aspect="equal")
            ax.set_title(f"frame {3 + j}: {label} n_e", fontsize=9)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
