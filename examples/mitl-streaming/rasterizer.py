"""Mesh fields -> hnode-style node frames with PCMS.

Two modes, both producing node-valued frames of shape grid.shape:

  conservative  L2 projection onto P1 on the covered pixels only (the target
                mesh conforms to the domain). Integrals are exact as
                sum(weights * frame); density uses a lumped mass matrix so it
                stays non-negative.
  point         pcms.Interpolator at every grid node, the analogue of the
                point sampling used to build the hnode training data.

See docs/mitl_pcms_data_loader_plan.md sections 3.3-3.6.
"""

import numpy as np

import pcms
from geometry import (compact, lagrange_space,
                      mesh_from_arrays, structured_triangles)

# Field name -> centering on the source mesh. Hardcoded to match the
# synthetic source; for EMPIRE this comes from the Exodus variable list.
FIELD_CENTERING = {
    "Ex": "node", "Ey": "node", "Bz": "elem",
    "n_e": "elem", "vx": "elem", "vy": "elem",
}
# hnode channel order (inferred from the bundled frames; ch5 = density).
CHANNELS = ("Ex", "Ey", "Bz", "vx", "vy", "n_e")
DENSITY_CHANNEL = CHANNELS.index("n_e")


def frame_channels(names):
    """hnode channel order restricted to the fields a source provides (the
    EMPIRE test run has no Ex, Ey, Bz)."""
    return tuple(c for c in CHANNELS if c in names)

CONSISTENT = pcms.MassMatrixType.Consistent
LUMPED = pcms.MassMatrixType.Lumped
# Signed fields: consistent (most accurate). Density and momentum density:
# lumped (non-negative) until the bounded projection lands.
MASS_TYPE = {"Ex": CONSISTENT, "Ey": CONSISTENT, "Bz": CONSISTENT,
             "n_e": LUMPED, "px": LUMPED, "py": LUMPED}


class GridRasterizer:
    """Maps source-mesh DOF vectors to (ny + 1, nx + 1) node frames.

    Operators are built once per (source centering, mass type) and reused for
    every step.
    """

    def __init__(self, src_mesh, grid, comm, mode="conservative"):
        if mode not in ("conservative", "point"):
            raise ValueError(f"unknown mode {mode!r}")
        self.grid, self.mode = grid, mode
        self._src = {"elem": lagrange_space(src_mesh, 0),
                     "node": lagrange_space(src_mesh, 1)}
        self._src_field = {k: s.create_field() for k, s in self._src.items()}
        self.src_areas = np.asarray(src_mesh.ask_sizes())

        nodes = grid.node_coords()
        x = grid.x0 + np.arange(grid.nx + 1) * grid.dx
        y = grid.y0 + np.arange(grid.ny + 1) * grid.dy
        _, tris = structured_triangles(x, y)
        if mode == "conservative":
            grid.check_step_aligned()
            keep = grid.inside(nodes[tris].mean(axis=1))
            self.mask = np.zeros(len(nodes), dtype=bool)
            self.mask[np.unique(tris[keep])] = True
        else:
            keep = np.ones(len(tris), dtype=bool)
            self.mask = grid.inside(nodes)
        tgt_points, tgt_tris, self._node_index = compact(nodes, tris, keep)
        self._tgt_mesh = mesh_from_arrays(tgt_points, tgt_tris, comm)
        self._tgt = lagrange_space(self._tgt_mesh, 1)
        self._tgt_field = self._tgt.create_field()
        self._ops = {}
        self.mask = self.mask.reshape(grid.shape)

        # Hat-function integrals over the domain only: the P1 lumped-mass
        # diagonal. A frame's integral is sum(weights * frame). The point
        # mode's target mesh also covers the step, so its own element sizes
        # would count the hats of nodes on the step beyond the domain.
        domain_tris = tris[grid.inside(nodes[tris].mean(axis=1))]
        w = np.zeros(len(nodes))
        for k in range(3):
            np.add.at(w, domain_tris[:, k], 0.5 * grid.dx * grid.dy / 3.0)
        self.weights = w.reshape(grid.shape)

    def _op(self, centering, mass_type):
        key = (centering, mass_type if self.mode == "conservative" else None)
        if key not in self._ops:
            src = self._src[centering]
            if self.mode == "conservative":
                op = pcms.OmegaHConservativeProjection(
                    src, self._tgt, mass_matrix_type=mass_type)
            else:
                op = pcms.Interpolator(
                    src, self._tgt,
                    pcms.OutOfBoundsPolicy(pcms.OutOfBoundsMode.FILL, 0.0))
            self._ops[key] = op
        return self._ops[key]

    def _scatter(self, values):
        frame = np.zeros(self.grid.shape[0] * self.grid.shape[1])
        frame[self._node_index] = values
        return frame.reshape(self.grid.shape)

    def map(self, values, centering, mass_type=CONSISTENT):
        field = self._src_field[centering]
        field.set_dof_holder_data(np.ascontiguousarray(values, np.float64))
        self._op(centering, mass_type).apply(field, self._tgt_field)
        frame = self._scatter(
            np.asarray(self._tgt_field.get_dof_holder_data())[:, 0])
        return np.where(self.mask, frame, 0.0)

    def integral(self, frame):
        return float(np.sum(self.weights * frame))

    def source_integral(self, elem_values):
        """Exact integral of a P0 source field."""
        return float(np.sum(self.src_areas * elem_values))


def make_frame(rast, fields, centering=None, dtype=np.float64):
    """One [C, ny + 1, nx + 1] frame in hnode channel order, C being the
    channels in frame_channels(fields) (6 for the synthetic source).

    centering maps field name -> "elem" / "node" (default FIELD_CENTERING).
    Kept in float64 by default so integrals stay exact; hnode stores float32
    frames, so cast at the disk/model boundary.

    Velocity is density weighted, R(n v) / R(n): exact momentum per pixel for
    P0 targets, approximate for P1 (plan section 3.6).
    """
    centering = FIELD_CENTERING if centering is None else centering

    def project(name, values, where):
        return rast.map(values, where, MASS_TYPE.get(name, LUMPED))

    channels = frame_channels(fields)
    n = fields["n_e"]
    out = {name: project(name, fields[name], centering[name])
           for name in ("Ex", "Ey", "Bz", "n_e") if name in channels}
    px = project("px", n * fields["vx"], "elem")
    py = project("py", n * fields["vy"], "elem")
    dens = out["n_e"]
    safe = np.where(dens > 0.0, dens, 1.0)
    out["vx"] = np.where(dens > 0.0, px / safe, 0.0)
    out["vy"] = np.where(dens > 0.0, py / safe, 0.0)
    return np.stack([out[c] for c in channels]).astype(dtype)
