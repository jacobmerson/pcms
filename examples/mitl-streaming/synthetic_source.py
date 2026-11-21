"""Synthetic stand-in for an EMPIRE MITL run: analytic fields Ex, Ey
(nodes), Bz, n_e, vx, vy (elements) on source_mesh().
"""

import numpy as np

from geometry import HEIGHT, STEP_X, STEP_Y, source_mesh
from rasterizer import FIELD_CENTERING

DENSITY_FLOOR = 1e12
BALL_PEAK = 1e17
FIELD_E0 = 1e6
FIELD_B0 = 1e-2
WAVE_SPEED = 0.5      # mm per step
DT = 1e-2             # time per stored step (hnode dt)


class SyntheticMITLSource:

    field_centering = FIELD_CENTERING

    def __init__(self, comm, num_steps=200, **mesh_options):
        self.mesh = source_mesh(comm, **mesh_options)
        self.num_steps = num_steps
        self.points = self.mesh.coords().reshape(-1, 2)
        self.tris = self.mesh.ask_elem_verts().reshape(-1, 3).astype(np.int64)
        self._verts = self.points
        self._cents = self.points[self.tris].mean(axis=1)

    def time(self, step):
        return step * DT

    def fields(self, step):
        t = float(step)
        vx_n, vy_n = self._verts[:, 0], self._verts[:, 1]
        cx, cy = self._cents[:, 0], self._cents[:, 1]

        front = WAVE_SPEED * t

        def pulse(x):
            return np.exp(-((x - front) / 4.0) ** 2)

        ex = FIELD_E0 * pulse(vx_n)
        ey = 0.3 * FIELD_E0 * pulse(vx_n) * np.sin(np.pi * vy_n / HEIGHT)
        bz = FIELD_B0 * pulse(cx)

        progress = min(t / 200.0, 1.0)
        bx = STEP_X + 1.0 + 6.0 * progress
        by = STEP_Y + 1.0
        ball = BALL_PEAK * progress * np.exp(
            -((cx - bx) ** 2 + (cy - by) ** 2) / (2 * 0.8 ** 2))
        islands = 0.0
        for phase in (0.0, np.pi):
            ang = 0.15 * t + phase
            ix = bx + 1.8 * np.cos(ang)
            iy = by + 1.2 * np.sin(ang)
            islands = islands + 0.5 * BALL_PEAK * progress * np.exp(
                -((cx - ix) ** 2 + (cy - iy) ** 2) / (2 * 0.12 ** 2))
        n_e = DENSITY_FLOOR + ball + islands

        omega = 2e6
        vx = 1e7 * progress - omega * (cy - by)
        vy = omega * (cx - bx)

        return {"Ex": ex, "Ey": ey, "Bz": bz, "n_e": n_e, "vx": vx, "vy": vy}
