"""The pretrained hnode plasma-gun surrogate behind the models.py interface.

    rollout(frames[3, 6, 101, 601], horizon) -> forecast[horizon, 6, 101, 601]

Needs the hnode repository (model code, plasma_gun_frames/stats.pt, the
checkpoint swin_plasma_e2e_redo_part_2.pth), einops and scikit-fem.
"""

import os
import sys

import numpy as np

LATENT_DIM = 64
DT = 1e-2                    # notebook dt: J = (z2 - z0) / DT
STEPS_PER_SAMPLE = 256
TRAINING_ROLLOUTS = 32
PADDED_SHAPE = (256, 768)
FRAMES_PER_LATENT_STEP = 2


class HNODEModel:

    name = "hnode"

    def __init__(self, hnode_dir, checkpoint=None, stats=None, z=0.0,
                 device=None):
        """Load the checkpoint.

        Args:
            z: voltage conditioning, -1 / 0 / 1 for 0.9 / 1.0 / 1.1 MV.
        """
        import torch
        self._torch = torch
        torch.set_default_dtype(torch.float64)
        if hnode_dir not in sys.path:
            sys.path.insert(0, hnode_dir)
        from hybrid_node_solver import HybridNODESolver
        from neural_networks import CrossAttnTransformer
        from swin_transformer_v2 import SwinDecoder, SwinEncoder

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = checkpoint or os.path.join(
            hnode_dir, "swin_plasma_e2e_redo_part_2.pth")
        stats = stats or os.path.join(hnode_dir, "plasma_gun_frames",
                                      "stats.pt")

        swin = dict(img_size=PADDED_SHAPE, patch_size=4, latent_dim=LATENT_DIM,
                    embed_dim=96, depths=(2, 2, 6, 2),
                    num_heads=(3, 6, 12, 24), window_size=(8, 8),
                    mlp_ratio=4, drop=0.0, attn_drop=0.0, drop_path_rate=0.0)
        self.encoder = SwinEncoder(in_chans=7, final_channels=64, **swin)
        self.decoder = SwinDecoder(out_chans=7, **swin)
        self.rhs = CrossAttnTransformer(
            problem_dim=LATENT_DIM, latent_dim=64 * 4, num_blocks=3,
            num_heads=4, z_dim=1, dropout_rate=0,
            activation=torch.nn.Tanh(), use_self_attention_z_block=True)

        # mmap: skips loading the ~5 GB optimizer state
        state = torch.load(checkpoint, map_location="cpu", mmap=True,
                           weights_only=False)
        for module, key in ((self.encoder, "encoder"),
                            (self.decoder, "decoder"),
                            (self.rhs, "nn_rhs_model")):
            module.load_state_dict(_strip_compile_prefix(state[key]))
            module.to(self.device).eval()
        del state

        n_t = STEPS_PER_SAMPLE // TRAINING_ROLLOUTS + 1
        self.latent_steps_per_rollout = n_t - 1
        self.solver = HybridNODESolver(
            self.rhs, DT * STEPS_PER_SAMPLE / TRAINING_ROLLOUTS, n_t,
            LATENT_DIM, TRAINING_ROLLOUTS, backprop_type="matrix_free",
            use_manual_jacobian=True, use_sparse_solver=False,
            use_cpu_solver=False, verbose=False, device=self.device)
        self.solver.include_mortars = False

        s = torch.load(stats, map_location="cpu")
        self.mask = s["mask"].double()                       # [H, W]
        self.mean = s["mean"].double()[:, None, None]        # [6, 1, 1]
        self.std = s["std"].double()[:, None, None]
        self.shape = tuple(self.mask.shape)
        self._padded_mask = self._pad(self.mask[None, None]).to(self.device)
        self.z = torch.tensor([[float(z)]], device=self.device)

    def _pad(self, x):
        F = self._torch.nn.functional
        return F.pad(x, (0, PADDED_SHAPE[1] - x.shape[-1],
                         0, PADDED_SHAPE[0] - x.shape[-2]))

    def _normalize(self, frames):
        torch = self._torch
        x = torch.as_tensor(np.asarray(frames), dtype=torch.float64)
        if tuple(x.shape[-2:]) != self.shape or x.shape[-3] != 6:
            raise ValueError(f"hnode needs [*, 6, {self.shape[0]}, "
                             f"{self.shape[1]}] frames, got {tuple(x.shape)}")
        x = (x - self.mean) / self.std
        x = torch.cat([x, self.mask.expand(x.shape[0], 1, *self.shape)], 1)
        x = self._pad(x).to(self.device)
        return torch.where(self._padded_mask > 0, x, 0.0)

    def rollout(self, frames, horizon):
        torch = self._torch
        with torch.no_grad():
            latents = self.encoder(self._normalize(frames))       # [3, L]
            batch = {"J_init": ((latents[2] - latents[0]) / DT)[None],
                     "u_init": latents[1][None], "z": self.z}
            # latent k is the frame FRAMES_PER_LATENT_STEP * k after the newest
            lead = np.arange(1, horizon + 1) / FRAMES_PER_LATENT_STEP
            needed = int(np.ceil(lead[-1])) + 1
            rollouts = -(-needed // self.latent_steps_per_rollout)
            _, u_hat = self.solver.rollout_batch(batch,
                                                 num_rollouts=rollouts)
            u_hat = u_hat[0]                                       # [K, L]
            lo = np.floor(lead).astype(int)
            w = torch.as_tensor(lead - lo, device=self.device)[:, None]
            hi = np.minimum(lo + 1, len(u_hat) - 1)
            z = (1 - w) * u_hat[lo] + w * u_hat[hi]
            out = self.decoder(z)[:, :6, :self.shape[0], :self.shape[1]]
            out = out.cpu() * self.std + self.mean
            out = torch.where(self.mask > 0, out, 0.0)
        return out.numpy()

    def train_step(self, frames, targets, leads, optimizer):
        """Take one optimizer step on streamed frames.

        Args:
            frames: three consecutive frames.
            targets: targets[k] is the frame leads[k] (even) after frames[-1].
        Returns:
            (loss, decoded targets in physical units).
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # the CPU flash attention kernel has no double backward
        with sdpa_kernel(SDPBackend.MATH):
            return self._train_step(frames, targets, leads, optimizer)

    def _train_step(self, frames, targets, leads, optimizer):
        torch = self._torch
        F = torch.nn.functional
        for module in (self.encoder, self.decoder, self.rhs):
            module.train()
        x = self._normalize(frames)                               # [3, 7, ...]
        y = self._normalize(targets)
        latents = self.encoder(x)
        batch = {"J_init": ((latents[2] - latents[0]) / DT)[None],
                 "u_init": latents[1][None], "z": self.z}
        saved = self.solver.num_training_rollouts
        self.solver.num_training_rollouts = 1
        try:
            _, u_hat = self.solver.evolve_batch(batch)
        finally:
            self.solver.num_training_rollouts = saved
        index = [lead // FRAMES_PER_LATENT_STEP for lead in leads]
        decoded = self.decoder(torch.cat([latents[:2], u_hat[0, index]]))
        decoded = torch.where(self._padded_mask > 0, decoded, 0.0)
        target = torch.cat([x[:2], y])
        covered = self._padded_mask.expand_as(decoded) > 0
        loss = F.l1_loss(decoded[covered], target[covered])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for module in (self.encoder, self.decoder, self.rhs):
            module.eval()
        with torch.no_grad():
            out = decoded[2:, :6, :self.shape[0], :self.shape[1]].cpu()
            out = torch.where(self.mask > 0, out * self.std + self.mean, 0.0)
        return float(loss.detach()), out.numpy()

    def parameters(self):
        for module in (self.encoder, self.decoder, self.rhs):
            yield from module.parameters()


class HNODEStreamTrainer:
    """Fine-tunes the model with one optimizer step per sample completed by an
    arriving frame; its predictions are made before the update.
    """

    def __init__(self, model, leads=(2, 4), lr=1e-6, max_steps=None):
        import torch
        self.model = model
        self.leads = list(leads)
        self.max_steps = max_steps
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.frames = []
        self.losses = []

    def receive(self, frame):
        """Returns None, or (loss, seconds, prediction of `frame`)."""
        import time
        self.frames.append(frame)
        newest = len(self.frames) - 1 - max(self.leads)
        if newest < 2 or (self.max_steps is not None
                          and len(self.losses) >= self.max_steps):
            return None
        t0 = time.perf_counter()
        loss, decoded = self.model.train_step(
            np.stack(self.frames[newest - 2:newest + 1]),
            np.stack([self.frames[newest + lead] for lead in self.leads]),
            self.leads, self.optimizer)
        self.losses.append(loss)
        return loss, time.perf_counter() - t0, decoded[-1]


def _strip_compile_prefix(state_dict):
    """The checkpoint was saved from torch.compile'd modules."""
    prefix = "_orig_mod."
    return {k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()}
