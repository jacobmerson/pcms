"""Replay of an EMPIRE MITL run from a joined Exodus II file (netCDF4).

Electron moments only, cell centred; coordinates in mm.
"""

import netCDF4
import numpy as np

from geometry import mesh_from_arrays

# Our field name -> EMPIRE element variable.
EMPIRE_FIELDS = {
    "n_e": "e- Number Density",
    "vx": "e- Average VelocityX",
    "vy": "e- Average VelocityY",
}
FIELD_CENTERING = {name: "elem" for name in EMPIRE_FIELDS}
LENGTH_SCALE = 1e3          # m -> mm


def _names(nc, var):
    if var not in nc.variables:
        return []
    return [bytes(row).rstrip(b"\0").decode().strip() for row in nc[var][:]]


class ExodusRun:

    field_centering = FIELD_CENTERING

    def __init__(self, path, comm=None, num_steps=None):
        """Open the run.

        Args:
            path: Exodus file.
            comm: Omega_h communicator to build the mesh on; None: no mesh.
            num_steps: steps to replay (default all).
        """
        self._nc = netCDF4.Dataset(path)
        self._nc.set_auto_mask(False)
        nc = self._nc
        self.points = LENGTH_SCALE * np.stack(
            [nc["coordx"][:], nc["coordy"][:]], 1)
        self._blocks = range(1, len(nc.dimensions["num_el_blk"]) + 1)
        self.tris = np.concatenate(
            [nc[f"connect{b}"][:] - 1 for b in self._blocks]).astype(np.int64)
        self._elem_vars = _names(nc, "name_elem_var")
        self._times = np.asarray(nc["time_whole"][:])
        stored = len(self._times)
        self.num_steps = stored if num_steps is None else min(num_steps,
                                                                stored)
        self.mesh = (mesh_from_arrays(self.points, self.tris, comm)
                     if comm is not None else None)

    def time(self, step):
        return float(self._times[step])

    def _elem_var(self, name, step):
        i = self._elem_vars.index(name) + 1
        return np.concatenate([self._nc[f"vals_elem_var{i}eb{b}"][step]
                               for b in self._blocks]).astype(np.float64)

    def fields(self, step):
        return {name: self._elem_var(var, step)
                for name, var in EMPIRE_FIELDS.items()}

    def proc_id(self):
        """EMPIRE's own element decomposition (rank per element)."""
        return self._elem_var("PROC_ID", 0).astype(np.int64)
