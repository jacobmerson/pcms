"""Cheap stand-ins for the hnode surrogate.

rollout(frames, horizon) -> forecast: frames is [3, C, H, W], oldest first;
forecast[j] is [C, H, W] for step j + 1 after the newest frame.
"""

import numpy as np


class PersistenceModel:
    """Forecast = newest frame. The baseline any surrogate must beat."""

    name = "persistence"

    def rollout(self, frames, horizon):
        return np.repeat(frames[-1:], horizon, axis=0)


class LinearExtrapolationModel:
    """Forecast = newest frame + (j + 1) * last difference."""

    name = "linear"

    def rollout(self, frames, horizon):
        rate = frames[-1] - frames[-2]
        steps = np.arange(1, horizon + 1, dtype=frames.dtype)
        return frames[-1][None] + steps[:, None, None, None] * rate[None]


class PooledTorchModel:
    """Untrained torch model on a pooled latent grid, exercising the
    surrogate's code path.
    """

    name = "pooled-torch"

    def __init__(self, pool=4, device=None):
        import torch
        self._torch = torch
        self.pool = pool
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")

    def rollout(self, frames, horizon):
        torch = self._torch
        F = torch.nn.functional
        x = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        size = x.shape[-2:]
        with torch.no_grad():
            z = F.avg_pool2d(x, self.pool, ceil_mode=True)      # encode
            u, du = z[1], 0.5 * (z[2] - z[0])                   # u, J
            steps = torch.arange(2, horizon + 2, device=self.device,
                                 dtype=z.dtype)
            latents = u[None] + steps[:, None, None, None] * du[None]
            out = F.interpolate(latents, size=size, mode="bilinear",
                                align_corners=False)            # decode
        return out.cpu().numpy()


MODELS = {cls.name: cls for cls in
          (PersistenceModel, LinearExtrapolationModel, PooledTorchModel)}
