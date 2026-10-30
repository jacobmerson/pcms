#!/usr/bin/env python3
"""Render a run_trainer.py record as a video (or a still of one step).

    python make_video.py train.npz train.mp4
    python make_video.py train.npz still.png --step 40

One video frame per streamed step: for each rasterization mode, the ingested
density, the model's prediction of it made before seeing it, and the
forecast made HORIZON steps earlier; below, the training loss, the
prediction error relative to persistence, and the ingested electron count
against the exact source integral. Needs matplotlib and, for video,
imageio-ffmpeg.
"""

import argparse

import numpy as np

TITLES = {"point": "point interpolation", "conservative": "PCMS conservative"}
COLORS = {"point": "#d95f02", "conservative": "#1b9e77"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("record")
    p.add_argument("output", help=".mp4 / .gif video, or .png for --step")
    p.add_argument("--step", type=int, default=None,
                   help="render only this step (a still)")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--hold", type=float, default=3.0,
                   help="seconds to hold the last frame")
    p.add_argument("--producers", type=int, default=8,
                   help="producer ranks, for the title")
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


class Renderer:

    def __init__(self, d, producers):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.d = d
        self.modes = [str(m) for m in d["modes"]]
        self.horizon = int(d["horizon"])
        self.n = len(d["step"])
        self.producers = producers
        truth = np.concatenate([d[f"{m}_truth"].ravel() for m in self.modes])
        self.vmax = float(np.nanmax(truth))
        self.vmin = self.vmax - 4.0
        self.cmap = matplotlib.colormaps["inferno"].copy()
        self.cmap.set_bad("#d9d9d9")
        self.extent = [float(v) for v in d["extent"]]
        self.persist = {}
        for m in self.modes:
            for key in ("err1", "errh"):
                e, p = d[f"{m}_{key}"], d[f"{m}_{key}_persist"]
                with np.errstate(invalid="ignore", divide="ignore"):
                    self.persist[m, key] = e / p
        # Optimizer steps done by the end of each received frame.
        self.steps_done = np.cumsum(np.isfinite(d[f"{self.modes[0]}_loss"])
                                    * int(d["train_steps"]))

    def figure(self, i):
        plt = self.plt
        d = self.d
        fig = plt.figure(figsize=(19.2, 10.8))
        gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.5],
                              hspace=0.45, wspace=0.5, left=0.04, right=0.97,
                              top=0.86, bottom=0.07)
        step, time = int(d["step"][i]), float(d["time"][i])
        fig.suptitle(
            f"EMPIRE MITL run streamed {self.producers} → 1 ranks over "
            f"the PCMS coupler (ADIOS2 SST); next-frame CNNs trained online",
            fontsize=18, y=0.975)
        fig.text(0.5, 0.915,
                 f"step {step}  ({time * 1e9:.1f} ns)     "
                 f"{int(self.steps_done[i])} optimizer steps per model     "
                 "every prediction is scored before the model sees the "
                 "frame", ha="center", fontsize=13, color="#444")
        rows = (("truth", "ingested frame"),
                ("pred1", "predicted before arrival (1 step)"),
                ("predh", f"forecast from {self.horizon} steps earlier"))
        for col, m in enumerate(self.modes):
            mask = d[f"{m}_mask"]
            for row, (key, label) in enumerate(rows):
                ax = fig.add_subplot(gs[row, 3 * col:3 * col + 3])
                img = np.where(mask, d[f"{m}_{key}"][i], np.nan)
                ax.imshow(img, origin="lower", cmap=self.cmap,
                          vmin=self.vmin, vmax=self.vmax,
                          extent=self.extent, aspect="equal",
                          interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                title = f"{TITLES[m]}: {label}"
                if key != "truth":
                    ratio = self.persist[m, "err1" if key == "pred1"
                                         else "errh"][i]
                    if np.isfinite(ratio):
                        title += f"   (error {ratio:.2f}× persistence)"
                    else:
                        title += "   (not yet)"
                ax.set_title(title, fontsize=11, color=COLORS[m]
                             if row == 0 else "black")
        self._plots(fig, gs, i)
        return fig

    def _plots(self, fig, gs, i):
        d = self.d
        steps = d["step"]
        upto = slice(0, i + 1)
        # Training loss per optimizer step.
        ax = fig.add_subplot(gs[3, 0:2])
        done = int(self.steps_done[i])
        for m in self.modes:
            losses = d[f"{m}_losses"]
            ax.plot(np.arange(1, done + 1), losses[:done], lw=0.8,
                    color=COLORS[m], alpha=0.35)
            k = 25
            if done >= k:
                smooth = np.convolve(losses[:done], np.ones(k) / k, "valid")
                ax.plot(np.arange(k, done + 1), smooth, lw=2,
                        color=COLORS[m], label=TITLES[m])
        ax.set_xlim(0, len(d[f"{self.modes[0]}_losses"]))
        all_losses = np.concatenate([d[f"{m}_losses"] for m in self.modes])
        if len(all_losses):
            ax.set_ylim(0, 1.05 * np.percentile(all_losses, 99.5))
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("training loss (masked L1)")
        ax.set_title("training loss", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        # Error relative to persistence.
        ax = fig.add_subplot(gs[3, 2:4])
        for m in self.modes:
            ax.plot(steps[upto], self.persist[m, "err1"][upto], "-",
                    color=COLORS[m], lw=1.2, alpha=0.6,
                    label=f"{TITLES[m]}, 1 step")
            ax.plot(steps[upto], self.persist[m, "errh"][upto], "-",
                    color=COLORS[m], lw=2.2,
                    label=f"{TITLES[m]}, {self.horizon} steps")
        ax.axhline(1.0, color="#888", ls="--", lw=1)
        ax.text(steps[-1], 1.0, "persistence ", ha="right", va="bottom",
                fontsize=9, color="#666")
        ax.set_xlim(steps[0], steps[-1])
        vals = np.concatenate([self.persist[m, k][np.isfinite(
            self.persist[m, k])] for m in self.modes for k in ("err1",
                                                               "errh")])
        if len(vals):
            ax.set_ylim(min(0.4, vals.min() * 0.95),
                        max(1.2, min(vals.max(), 2.0) * 1.02))
        ax.set_xlabel("streamed step")
        ax.set_ylabel("error / persistence error")
        ax.set_title("prediction error (lower is better)", fontsize=12)
        ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.9)
        # Electron count.
        ax = fig.add_subplot(gs[3, 4:6])
        src = d["source_count"]
        for m in self.modes:
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = 100 * (d[f"{m}_count"] / src - 1)
            ax.plot(steps[upto], rel[upto], "-o", ms=3, color=COLORS[m],
                    label=TITLES[m])
        ax.axhline(0.0, color="#888", ls="--", lw=1)
        ax.set_xlim(steps[0], steps[-1])
        rel_all = np.concatenate([100 * (d[f"{m}_count"][1:] / src[1:] - 1)
                                  for m in self.modes])
        rel_all = rel_all[np.isfinite(rel_all)]
        if len(rel_all):
            lim = min(max(np.abs(rel_all).max() * 1.1, 1.0), 50.0)
            ax.set_ylim(-lim, lim)
        ax.set_xlabel("streamed step")
        ax.set_ylabel("electron count error (%)")
        ax.set_title("ingested electrons vs exact source", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")

    def rgb(self, i, dpi):
        fig = self.figure(i)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        self.plt.close(fig)
        return img


class HNODERenderer:
    """A run_trainer.py --model hnode record: streamed frame, the hnode
    prediction of it from the forward pass before the update, and the loss
    per fine-tuning step."""

    def __init__(self, d, producers):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.d = d
        self.n = len(d["step"])
        self.producers = producers
        self.vmax = float(np.nanmax(d["truth"]))
        self.vmin = self.vmax - 5.0
        self.cmap = matplotlib.colormaps["inferno"].copy()
        self.cmap.set_bad("#d9d9d9")
        self.extent = [float(v) for v in d["extent"]]

    def figure(self, i):
        plt = self.plt
        d = self.d
        fig = plt.figure(figsize=(19.2, 10.8))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.4,
                              wspace=0.15, left=0.04, right=0.97, top=0.86,
                              bottom=0.07)
        loss = d["loss"]
        done = int(np.isfinite(loss[:i + 1]).sum())
        fig.suptitle(
            f"Pretrained hnode surrogate fine-tuned on a stream: "
            f"{self.producers} → 1 ranks over the PCMS coupler "
            f"(ADIOS2 SST)", fontsize=18, y=0.975)
        fig.text(0.5, 0.915,
                 f"step {int(d['step'][i])}     {done} optimizer steps     "
                 "synthetic 6-channel source (the code path, not physics)",
                 ha="center", fontsize=13, color="#444")
        mask = d["mask"]
        panels = (("truth", "streamed frame (log10 n_e)"),
                  ("pred", f"hnode prediction, {int(d['lead'])} steps "
                           "ahead, before this update"),
                  ("persist", "persistence"))
        slots = (gs[0, 0], gs[0, 1], gs[1, 0])
        for slot, (key, label) in zip(slots, panels):
            ax = fig.add_subplot(slot)
            ax.imshow(np.where(mask, d[key][i], np.nan), origin="lower",
                      cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
                      extent=self.extent, aspect="equal",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=12)
        ax = fig.add_subplot(gs[1, 1])
        t = d["train_time"][np.isfinite(d["train_time"])]
        ax.axis("off")
        ax.text(0.02, 0.6,
                f"one optimizer step: {np.mean(t):.0f} s on CPU\n"
                "(encoder, one latent rollout with implicit backprop,\n"
                "decoder; float64, as trained)" if len(t) else "",
                fontsize=13, va="center")
        ax = fig.add_subplot(gs[2, :])
        steps = np.arange(1, done + 1)
        ax.plot(steps, loss[np.isfinite(loss)][:done], "-o",
                color=COLORS["conservative"], lw=2)
        ax.set_xlim(0.5, max(np.isfinite(loss).sum(), 1) + 0.5)
        finite = loss[np.isfinite(loss)]
        if len(finite):
            pad = 0.1 * max(np.ptp(finite), 1e-3 * finite.max())
            ax.set_ylim(finite.min() - pad, finite.max() + pad)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("training loss (masked L1)")
        ax.set_title("hnode fine-tuning loss", fontsize=12)
        return fig

    rgb = Renderer.rgb


def main():
    args = parse_args()
    d = dict(np.load(args.record))
    if str(d.get("model", "")) == "hnode":
        r = HNODERenderer(d, args.producers)
    else:
        r = Renderer(d, args.producers)
    if args.step is not None:
        i = int(np.flatnonzero(d["step"] == args.step)[0])
        fig = r.figure(i)
        fig.savefig(args.output, dpi=args.dpi)
        print(f"wrote {args.output}")
        return
    import imageio.v2 as imageio
    kwargs = {"fps": args.fps}
    if args.output.endswith(".mp4"):
        kwargs.update(codec="libx264", quality=8, macro_block_size=8,
                      pixelformat="yuv420p")
    with imageio.get_writer(args.output, **kwargs) as w:
        for i in range(r.n):
            img = r.rgb(i, args.dpi)
            w.append_data(img)
            print(f"\rframe {i + 1}/{r.n}", end="", flush=True)
        for _ in range(int(args.hold * args.fps)):
            w.append_data(img)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
