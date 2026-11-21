"""Mesh fields to node frames with PCMS: conservative projection onto P1
on the covered pixels, or point interpolation at every grid node.
"""

import numpy as np

import pcms
from geometry import (compact, lagrange_space, mesh_from_arrays,
                      mesh_geometry, structured_triangles)

# field name -> centering of the synthetic source
FIELD_CENTERING = {
    "Ex": "node", "Ey": "node", "Bz": "elem",
    "n_e": "elem", "vx": "elem", "vy": "elem",
}
# hnode channel order, inferred from its bundled frames
CHANNELS = ("Ex", "Ey", "Bz", "vx", "vy", "n_e")
DENSITY_CHANNEL = CHANNELS.index("n_e")


def frame_channels(names):
    """Return the hnode channels, in order, that `names` provides."""
    return tuple(c for c in CHANNELS if c in names)

CONSISTENT = pcms.MassMatrixType.Consistent
LUMPED = pcms.MassMatrixType.Lumped
# lumped keeps density and momentum density non-negative
MASS_TYPE = {"Ex": CONSISTENT, "Ey": CONSISTENT, "Bz": CONSISTENT,
             "n_e": LUMPED, "px": LUMPED, "py": LUMPED}


class GridRasterizer:
    """Maps source-mesh DOF values to (ny + 1, nx + 1) node frames; transfer
    operators are built once and reused.
    """

    def __init__(self, src_mesh, grid, comm, mode="conservative"):
        if mode not in ("conservative", "point"):
            raise ValueError(f"unknown mode {mode!r}")
        self.grid, self.mode = grid, mode
        src_geometry = mesh_geometry(src_mesh)
        self._src = {"elem": lagrange_space(src_geometry, 0),
                     "node": lagrange_space(src_geometry, 1)}
        self._src_field = {k: s.create_function()
                           for k, s in self._src.items()}
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
        self._tgt = lagrange_space(mesh_geometry(self._tgt_mesh), 1)
        self._tgt_field = self._tgt.create_function()
        self._ops = {}
        self.mask = self.mask.reshape(grid.shape)

        # domain only: the point-mode target mesh also covers the step
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
                op = pcms.create_conservative_projection(
                    src, self._tgt, mass_matrix_type=mass_type)
            else:
                op = pcms.create_interpolator(
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
            self._tgt_field.get_dof_holder_data().get_values()[:, 0])
        return np.where(self.mask, frame, 0.0)

    def integral(self, frame):
        return float(np.sum(self.weights * frame))

    def source_integral(self, elem_values):
        """Exact integral of a P0 source field."""
        return float(np.sum(self.src_areas * elem_values))


def make_frame(rast, fields, centering=None, dtype=np.float64):
    """Return one [C, ny + 1, nx + 1] frame in hnode channel order.

    Args:
        rast: GridRasterizer.
        fields: field name -> source DOF values.
        centering: field name -> "elem" / "node" (default FIELD_CENTERING).
        dtype: frame dtype.
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
