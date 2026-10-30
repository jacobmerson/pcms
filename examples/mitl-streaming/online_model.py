"""Online training of a small next-frame model on streamed frames.

The model learns frame t + 1 from frames t - 1 and t while the run streams
in. Each received frame is first scored (prequential evaluation: the model
predicts it before training on it, so every score is on unseen data), then
added to a replay buffer, then the model takes a few optimizer steps on
samples drawn from the buffer, half from the newest frames.

Features: density as log10(n_e + floor), shifted and scaled to about
[0, 1]; every other channel divided by the largest |value| seen so far in
that channel (a running scale, so nothing has to be known up front). The
network predicts the change from the newest frame and its last layer starts
at zero, so an untrained model is persistence and any skill is learned.
"""

import collections
import time

import numpy as np

DENSITY_FLOOR = 1e12
DENSITY_DECADES = 7.0


class Featurizer:

    def __init__(self, channels, density_channel):
        self.channels = channels
        self.density_channel = density_channel
        self.scale = np.zeros(len(channels))

    def observe(self, frame, mask):
        for c in range(len(self.channels)):
            if c != self.density_channel:
                self.scale[c] = max(self.scale[c],
                                    float(np.abs(frame[c][mask]).max()))

    def __call__(self, frame):
        out = np.empty(frame.shape, dtype=np.float32)
        for c in range(frame.shape[0]):
            if c == self.density_channel:
                out[c] = (np.log10(np.maximum(frame[c], 0.0) + DENSITY_FLOOR)
                          - np.log10(DENSITY_FLOOR)) / DENSITY_DECADES
            else:
                out[c] = frame[c] / (self.scale[c] or 1.0)
        return out

    def log_density(self, features):
        """Feature density channel -> log10(n_e + floor)."""
        return (features[self.density_channel] * DENSITY_DECADES
                + np.log10(DENSITY_FLOOR))


def make_net(channels, width=16):
    """Small two-level U-Net: [B, 2C + 1, H, W] -> change [B, C, H, W]."""
    import torch
    from torch import nn
    F = nn.functional

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1 = block(2 * channels + 1, w)
            self.e2 = block(w, 2 * w, stride=2)
            self.e3 = block(2 * w, 3 * w, stride=2)
            self.d2 = block(5 * w, 2 * w)
            self.d1 = block(3 * w, w)
            self.out = nn.Conv2d(w, channels, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, x):
            x1 = self.e1(x)
            x2 = self.e2(x1)
            x3 = self.e3(x2)
            up = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear",
                               align_corners=False)
            y = self.d2(torch.cat([up, x2], 1))
            up = F.interpolate(y, size=x1.shape[-2:], mode="bilinear",
                               align_corners=False)
            return self.out(self.d1(torch.cat([up, x1], 1)))

    return Net()


def masked_l1(a, b, mask):
    """Mean |a - b| over the covered pixels and all channels (numpy)."""
    return float(np.abs(a - b)[:, mask].mean())


class OnlineTrainer:
    """One model trained on one stream of frames (one rasterization)."""

    def __init__(self, channels, density_channel, mask, steps_per_frame=20,
                 batch_size=4, lr=1e-3, unroll=2, horizon=5, seed=0,
                 recent=8, width=16, crop=(64, 192)):
        import torch
        self._torch = torch
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.feat = Featurizer(channels, density_channel)
        self.mask = mask
        self._mask_t = torch.as_tensor(mask, dtype=torch.float32)
        self.net = make_net(len(channels), width)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.steps_per_frame = steps_per_frame
        self.batch_size = batch_size
        self.unroll = unroll
        self.horizon = horizon
        self.recent = recent
        self.crop = crop              # training patch (h, w); None: whole
        self.raw = []                 # received frames, in order
        self.optimizer_steps = 0
        self.losses = []              # every optimizer step
        self._forecasts = {}          # target index -> (features, lead)

    # Model.
    def _step(self, prev, cur, mask=None):
        """prev, cur: [B, C, H, W] feature tensors -> next. mask: [B, 1, H,
        W] (default: the whole frame's)."""
        torch = self._torch
        if mask is None:
            mask = self._mask_t[None, None]
        mask = mask.expand(cur.shape[0], 1, *cur.shape[-2:])
        nxt = cur + self.net(torch.cat([prev, cur, mask], 1))
        return nxt * mask

    def _predict(self, feats, steps):
        """Roll out `steps` frames after feats[-1] (numpy in and out)."""
        torch = self._torch
        prev = torch.as_tensor(feats[-2])[None]
        cur = torch.as_tensor(feats[-1])[None]
        out = []
        with torch.no_grad():
            for _ in range(steps):
                prev, cur = cur, self._step(prev, cur)
                out.append(cur[0].numpy())
        return out

    # Stream.
    def receive(self, frame):
        """Score, store and train on one received frame; returns a dict of
        per-step results (errors are in feature units, masked L1)."""
        t = len(self.raw)
        self.feat.observe(frame, self.mask)
        self.raw.append(frame)
        result = {"index": t, "err1": np.nan, "err1_persist": np.nan,
                  "errh": np.nan, "errh_persist": np.nan, "pred1": None,
                  "predh": None, "loss": np.nan, "train_time": 0.0}
        truth = self.feat(frame)
        result["truth"] = truth
        if t >= 2:
            prev = [self.feat(self.raw[t - 2]), self.feat(self.raw[t - 1])]
            pred = self._predict(prev, 1)[0]
            result["pred1"] = pred
            result["err1"] = masked_l1(pred, truth, self.mask)
            result["err1_persist"] = masked_l1(prev[-1], truth, self.mask)
        if t in self._forecasts:
            pred, lead = self._forecasts.pop(t)
            result["predh"] = pred
            result["errh"] = masked_l1(pred, truth, self.mask)
            result["errh_persist"] = masked_l1(
                self.feat(self.raw[t - lead]), truth, self.mask)
        t0 = time.perf_counter()
        losses = [self._train_step() for _ in range(self.steps_per_frame)]
        losses = [x for x in losses if x is not None]
        result["train_time"] = time.perf_counter() - t0
        if losses:
            result["loss"] = float(np.mean(losses))
        if t >= 1:
            # Forecast made now (after training on t), scored on arrival.
            feats = [self.feat(self.raw[t - 1]), truth]
            self._forecasts[t + self.horizon] = (
                self._predict(feats, self.horizon)[-1], self.horizon)
        return result

    def _train_step(self):
        torch = self._torch
        # Sample k: inputs k - 1, k; targets k + 1 .. k + u.
        u = min(self.unroll, len(self.raw) - 2)
        if u < 1:
            return None
        last = len(self.raw) - 1 - u
        first = 1
        n_recent = self.batch_size // 2
        lo = max(first, last - self.recent + 1)
        ks = np.concatenate([
            self.rng.integers(lo, last + 1, n_recent),
            self.rng.integers(first, last + 1, self.batch_size - n_recent)])
        seq = np.stack([[self.feat(self.raw[k + j]) for j in range(-1, u + 1)]
                        for k in ks])                 # [B, u + 2, C, H, W]
        mask = np.broadcast_to(self.mask, (len(ks), 1) + self.mask.shape)
        if self.crop:
            # Random patches: the network is convolutional, so it trains on
            # patches several times faster and is applied to whole frames.
            h, w = self.crop
            H, W = self.mask.shape
            y0 = self.rng.integers(0, H - h + 1, len(ks))
            x0 = self.rng.integers(0, W - w + 1, len(ks))
            seq = np.stack([q[..., y:y + h, x:x + w]
                            for q, y, x in zip(seq, y0, x0)])
            mask = np.stack([q[..., y:y + h, x:x + w]
                             for q, y, x in zip(mask, y0, x0)])
        seq = torch.as_tensor(seq)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        covered = mask.expand(-1, seq.shape[2], -1, -1) > 0
        prev, cur = seq[:, 0], seq[:, 1]
        loss = 0.0
        for j in range(u):
            prev, cur = cur, self._step(prev, cur, mask)
            loss = loss + (cur - seq[:, j + 2]).abs()[covered].mean()
        loss = loss / u
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.optimizer_steps += 1
        loss = float(loss.detach())
        self.losses.append(loss)
        return loss


class OnlineRecord:
    """Per-step arrays for the video, kept as float32 log density maps."""

    def __init__(self):
        self.rows = collections.defaultdict(list)

    def add(self, prefix, trainer, result):
        r = self.rows
        for key in ("err1", "err1_persist", "errh", "errh_persist", "loss",
                    "train_time"):
            r[f"{prefix}_{key}"].append(result[key])
        shape = trainer.mask.shape
        for key in ("truth", "pred1", "predh"):
            f = result[key]
            img = (np.full(shape, np.nan, np.float32) if f is None else
                   trainer.feat.log_density(f).astype(np.float32))
            r[f"{prefix}_{key}"].append(img)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}
