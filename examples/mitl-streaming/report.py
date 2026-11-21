"""Console summary and figure for a demo run (loopback or two-process)."""

import numpy as np


def summarize(consumer, records, num_steps, dropped, elapsed, model_name):
    counts = np.array([(r.count, r.source_count) for r in records])
    cons = np.abs(counts[:, 0] - counts[:, 1]) / np.maximum(
        np.abs(counts[:, 1]), np.finfo(float).tiny)
    errs = np.array([r.density_error for r in records])
    print(f"received {len(records)} of {num_steps} steps "
          f"(dropped {dropped}, gaps {consumer.gaps}) in "
          f"{elapsed:.2f} s, "
          f"{elapsed / max(len(records), 1) * 1e3:.1f} ms/step")
    nonzero = counts[:, 1] != 0
    if nonzero.any():
        print(f"electron count vs source: max rel err "
              f"{cons[nonzero].max():.2e}")
    if np.isfinite(errs).any():
        print(f"{model_name} forecast density rel L2: "
              f"median {np.nanmedian(errs):.3f}, max {np.nanmax(errs):.3f} "
              f"over {np.isfinite(errs).sum()} steps, "
              f"{len(consumer.anchors)} anchors")


def plot(path, consumer, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(consumer.frames)
    mask = consumer.rast.mask
    dens_all = [img[consumer.density_channel][mask]
                for truth, pred in consumer.frames.values()
                for img in (truth, pred) if img is not None]
    vmax = np.log10(max(max(d.max() for d in dens_all), 1.0))
    vmin = vmax - 5.0
    fig, axes = plt.subplots(len(steps) + 1, 2, figsize=(11, 1.6 * len(steps)
                                                         + 2.4))
    for row, step in enumerate(steps):
        truth, pred = consumer.frames[step]
        for col, (img, label) in enumerate(((truth, "streamed (PCMS)"),
                                            (pred, "forecast"))):
            ax = axes[row, col]
            ax.set_axis_off()
            if img is None:
                ax.set_title(f"step {step}: no forecast", fontsize=9)
                continue
            dens = np.where(mask, img[consumer.density_channel], np.nan)
            ax.imshow(np.log10(np.maximum(dens, 10.0 ** vmin)),
                      origin="lower", cmap="inferno", vmin=vmin, vmax=vmax,
                      aspect="equal")
            ax.set_title(f"step {step}: {label} log10 n_e", fontsize=9)
    ax = plt.subplot2grid((len(steps) + 1, 2), (len(steps), 0), colspan=2,
                          fig=fig)
    axes[len(steps), 0].remove()
    axes[len(steps), 1].remove()
    s = [r.step for r in records]
    ax.plot(s, [r.source_count for r in records], label="source (exact)")
    ax.plot(s, [r.count for r in records], "--", label="ingested frame")
    ax.plot(s, [r.forecast_count for r in records], ":", label="forecast")
    ax.set_xlabel("step")
    ax.set_ylabel("electron count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")
