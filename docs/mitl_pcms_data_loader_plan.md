# PCMS-based data loader for the hnode MITL (plasma gun) surrogate

Status: **draft for review**. Last updated 2026-10-27. One EMPIRE run (electron
moments only, 51 steps) is in the PCMS test data repo and is described in
section 8.1; the variable names and centering in sections 3-5 predate it and
are placeholders. The example code in section 5 is an untested sketch
against the current Python bindings. The streaming demo through step 2 and
the hnode adapter are implemented; see 8.9 for what exists and what was
learned.

**For implementers:** the streaming demo (section 8) is the active track.
Read section 0 (getting started), then 8.1 (what exists, test data), 8.6
(tasks) and 8.7 (sequence). Sections 2-7 are background for the offline
converter and training study; skim as needed.

## 0. Getting started

Everything referenced here is either in the PCMS repository, in a public
repository listed below, or explicitly marked as not yet available.

**Repositories and data**

| What | Where |
|---|---|
| PCMS | https://github.com/SCOREC/pcms, branch `develop`. This doc is `docs/mitl_pcms_data_loader_plan.md`. |
| Streaming prototype | `examples/mitl-streaming/` in the same PCMS branch as this doc (see its `README.md`). |
| EMPIRE test run | https://github.com/SCOREC/pcms_testcases, file `exodus/flow/flow.exo` (Git LFS). `git clone` the repo, then `git lfs pull --include exodus/flow/flow.exo`. Contents in 8.1. |
| hnode surrogate | https://github.com/PIMILab/hnode: notebook `04_plasma_gun.ipynb`, bundled frames `plasma_gun_frames/example_traj`, and the pretrained checkpoint `swin_plasma_e2e_redo_part_2.pth`. The full clone is about 11 GB. |
| Paper | Kinch et al., arXiv:2604.21101 (Fig. 3, Appendix E.4). |
| Full EMPIRE dataset (12 runs, with fields) | Not available yet; see open question 1. |

**Build**

PCMS with Omega_h, PETSc, MeshFields and Python enabled
(`-DPCMS_ENABLE_OMEGA_H=ON -DPCMS_ENABLE_PETSC=ON -DPCMS_ENABLE_MESHFIELDS=ON
-DPCMS_ENABLE_Python=ON`); full dependency recipes are in `README.md` and the
Spack environment in `spack/`. For the streaming work, ADIOS2 must be built
with SST and MPI (`ADIOS2_HAVE_SST`, `ADIOS2_HAVE_MPI` in its CMake config),
and `mpi4py` must be installed against the same MPI as PCMS
(`MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py`, using that
MPI's `mpicc`).
At runtime, the built `pcms` module and `PyOmega_h` must be on `PYTHONPATH`.
Project conventions (formatting, naming, test macros) are in
`CONTRIBUTING.md`.

**First check that the prototype runs**

```bash
cd examples/mitl-streaming
python -m unittest discover -s tests -t .
python run_loopback_demo.py --steps 120
```

Both must pass before starting S1; they exercise every piece except the PCMS
transport.

**Not yet available (do not block on these)**

- The bounded (Farrell 2009) projection (3.4) is not on `develop`; use
  lumped for density in the meantime.
- The earlier ML-ingestion study cited in 6.1 and 9 is not in the
  repository; its numbers are context only.

## 1. Context

Kinch et al., *A Hybridizable Neural Time Integrator for Stable
Autoregressive Forecasting* (arXiv:2604.21101), Fig. 3 / Appendix E.4: a
surrogate of the 2D inner MITL (magnetically insulated transmission line) of
the Z-machine. Code: https://github.com/PIMILab/hnode, notebook
`04_plasma_gun.ipynb` ("Horseshoe2D").

- **Simulation:** EMPIRE-PIC, 55,065 triangles, 59,958 steps over 10 ns.
  12 runs = 3 peak voltages (0.9 / 1.0 / 1.1 MV) x 4 seeds.
- **Stored data:** every 5th step, giving 11,991 snapshots per run. Particles
  averaged to per-cell electron density and velocity, plus Ex, Ey, Bz.
  1.1 TB total.
- **Model:** 6 normalized channels + binary mask, zero-padded to 256x768,
  then Swin-V2 encoder (64-dim latent), HNODE latent dynamics conditioned on
  voltage (z in {-1, 0, 1}), Swin decoder. Masked L1 loss.
- **Interface hnode reads** (`Horseshoe2DDataset`):
  ```
  <root>/stats.pt                       {mask [H,W], mean [6], std [6]}
  <root>/<run>/frames/frame_%06d.pt     float32 [6, H, W]
  ```

Goal: replace their (unpublished) mesh-to-grid preprocessing with a PCMS
conservative projection that writes the same layout, so hnode needs few or
no changes.

## 2. Current preprocessing (inferred)

The preprocessing script is not in the repo. From the paper ("uniformly
sampled onto a regular 600 x 100 grid") and the 10 bundled frames
(`plasma_gun_frames/example_traj`):

- **Point samples at grid nodes.** Frames are `[6, 101, 601]`, the vertex
  count of a 600x100 grid. This is what a ParaView/VTK resample-to-image or
  probe filter produces.
- **Channels 0-4 are interpolated**, not piecewise constant: about 97% of
  in-domain values are unique and only about 2% of horizontal neighbours are
  equal.
- **Channel 5 is density** (named in the notebook). The bundled frames are
  too early to tell how it was sampled: they are mostly the 1e12 floor, with
  35 distinct values.
- **Channel order guess:** `[Ex, Ey, Bz, vx, vy, n_e]`. Channels 0-2 are zero
  at t=0; channels 3-4 have std around 1e7 (velocity-like). Must be confirmed
  against the Exodus variable names.
- **Binary mask with boundary holes.** Row 0 is missing columns 438-439, 493
  and 574: nodes exactly on the wall were classified as outside, the classic
  point-in-mesh failure.
- **Geometry check:** the mask step at column 420 = 35 mm matches Fig. 6
  (50 mm long, last 15 mm expanded), so dx = 50/600 mm and the grid spans the
  domain bounding box.

Consequences:
- Electron count is not preserved.
- Where mesh cells are smaller than the grid spacing (likely near the
  emitting cathode), one cell is picked per node and the rest are ignored
  (aliasing).
- PIC shot noise passes through unchanged.

## 3. Design

The loader is an **offline converter**, not an online `Dataset`. hnode's
`DataLoader` forks 4 worker processes, which is a bad fit for Kokkos/PETSc.
Training also re-reads frames over about 145k optimizer steps, so it is
better to pay the conversion cost once.

```
Exodus run
  └─ ExodusRun          mesh once; fields(step) -> {name: ndarray}
       └─ GridRasterizer  target mesh + cached PCMS operators
            └─ channel recipe
                 └─ FrameWriter  <run>/frames/frame_%06d.pt, stats.pt
```

Location: `examples/python-api-example/mitl_exodus_to_frames.py`. Later it
may share a generalized rasterizer with other mesh-to-tensor examples and
become a general `load_exodus` Python loader (planned separately).

### 3.1 ExodusRun (reader)

The Omega_h builds used so far have `Omega_h_USE_SEACASExodus=OFF`, so the
`omega_h.read_mesh_exodus` / `read_exodus_*_fields` bindings are not
compiled (check yours). There are two interchangeable backends:

- **SEACAS:** rebuild Omega_h with Exodus support. Clean long-term path.
- **netCDF4:** read connectivity, coordinates and `vals_elem_var*` /
  `vals_nod_var*` directly, and build the Omega_h mesh through a gmsh round
  trip (`omega_h.read_mesh_gmsh`). Per-step reads are array slices, which is
  likely faster than populating tags every step. No rebuild needed.
  **Section 5 uses this backend.**

The test file (8.1) is a single joined file with all steps, so the netCDF4
backend reads it directly (`scipy.io.netcdf_file` also works: it is netCDF
classic 64-bit offset). The full dataset's layout is still unknown.

### 3.2 GridRasterizer (PCMS)

- One source space per centering present: P0 for cell moments, P1 for nodal
  E/B (if EMPIRE wrote them at nodes).
- One cached operator per (source centering, mass-matrix type).
- Operators are built once per mesh pair. Construction caches intersection,
  quadrature and mass-matrix setup; `apply()` assembles a right-hand side and
  solves with the cached KSP (`omega_h_conservative_projection.hpp:24-33`).
  All 12 runs likely share one mesh, so setup happens once in total.
- `Apply` is scalar-only (`omega_h_conservative_projection.cpp:33`), so
  vector components are mapped one at a time.
- Possible later speedup: export the operator as a sparse matrix and map all
  fields/steps in one product. Only worth it if profiling shows `apply`
  dominating; Exodus I/O likely costs more.

### 3.3 Target discretization

**Recommended: P1 on a domain-matched 601x101 node grid.**

| | Current (point samples) | P0 cell averages | P1, domain-matched |
|---|---|---|---|
| Shape | 601x101 | 600x100 | 601x101 |
| Electron count | Not preserved | Exact, plain pixel sum | Exact, weighted sum |
| Small-cell aliasing | Yes | No | No |
| Sharp features | Point values | Pixel-averaged | Depends on mass matrix (3.4) |
| Positivity | Yes | Yes | Depends on mass matrix (3.4) |
| Momentum | Not preserved | Exact via R(n v)/R(n) | Exact only if storing n v (3.6) |
| Existing checkpoint | Trained on it | Retrain | Fine-tune |

P1 keeps the current frame shape and node-based mask. Where the mesh is
coarser than the grid it is nearly identical to interpolated sampling; where
the mesh is finer it averages instead of aliasing.

**Conservation under P1** means the integral of the projected field,
`sum_i w_i u_i`, equals the source integral. Here `w_i` is the integral of
node i's hat function, i.e. the mass-matrix row sums. Interior nodes get
dx*dy and edge/corner nodes get less; the two-triangles-per-pixel diagonal
makes corner weights asymmetric. `w` is stored in `stats.pt`, and downstream
totals and centre of mass must use it.

**P0 alternative:** 600x100 cell averages with `MassMatrixType.Lumped`. For
P0 this is identical to consistent and is an exact diagonal solve; it also
avoids a PETSc crash seen with the default ILU preconditioner on
~120k-row pixel-mesh mass matrices (SIGBUS in the KokkosKernels ILU numeric
phase). If a large consistent solve crashes, set
`PETSC_OPTIONS="-ksp_type cg -pc_type jacobi"` before importing `pcms`. It fits the 256x768 padding but requires
retraining.

**If more than one resolution is wanted, prefer P0.** On nested grids, P0
cell averages coarsen exactly by average pooling, so PCMS runs once at
600x100 and every coarser dataset is free (section 6.2). Lumped P1 does not
have this property.

### 3.4 Mass matrix (P1)

| | Accuracy | Density >= 0 | Cost per apply | Availability |
|---|---|---|---|---|
| Consistent | Best in L2, sharp | No: overshoots at fronts (density spans ~1e12 to ~1e17) | CG solve, tens of iterations (~60k rows, well conditioned) | develop |
| Lumped | Smears ~1 pixel | Yes | Diagonal divide | develop |
| Bounded (Farrell 2009) | Close to consistent | Yes, hard bound, conservative | Consistent + repair sweeps | Not on develop yet (unpublished work in progress) |

Plan:
- E, B (no sign constraint): consistent.
- Density, momentum: bounded once it is recovered and merged; lumped in the
  meantime.
- The KSP tolerance is already tight on develop
  (`PCMS_PROJECTION_KSP_RTOL=1e-14`, `CMakeLists.txt:65`), so consistent
  solves do not limit conservation.

### 3.5 Domain boundary

If the target covers the full bounding box, nodes outside the domain are
pinned at zero. A consistent projection then rings at the walls, and wall
nodes see only part of their support, so their values are diluted.

Instead, build the target mesh **only from covered pixels** so it matches
the domain. The step at column 420 lies on a grid line; if the other walls
do too, there is no dilution, and the mask is simply "is a target-mesh
node". That also removes the row-0 holes. Walls between grid lines leave a
thin layer of partially covered pixels. For those, per-node coverage (the
lumped projection of 1) is stored and values are divided by it.

To verify from the Exodus mesh: wall positions relative to grid lines.

### 3.6 Channel recipe

`R` = conservative projection onto the target space.

| Channel | Computation | Conserved |
|---|---|---|
| Ex, Ey, Bz | `R(E)`, `R(B)` (consistent) | Field integrals |
| n_e | `R(n)` (bounded / lumped) | Electron count |
| momentum x, y (if retraining) | `R(n vx)`, `R(n vy)` | Momentum |
| vx, vy (checkpoint-compatible) | `R(n vx) / R(n)` per node | Approximate only |

Under P1 the ratio does not conserve momentum, because the product of two P1
fields is not P1. Storing momentum density is linear in the data, so it is
exactly conserved, and normalization handles the scale. Keep velocity only
to stay compatible with the existing checkpoint.

### 3.7 FrameWriter and stats

- `torch.save(float32 [C, ny+1, nx+1])` as `frame_%06d.pt`, hnode's naming.
- `stats.pt`: binary mask and per-channel mean/std from a streaming pass over
  **training runs only**, plus `w` (3.3) and coverage (3.5). Extra keys are
  ignored by the existing `Horseshoe2DDataset`.
- One process per run; the 12 runs convert in parallel, no MPI needed.
- Omega_h lifetime gotcha: `OmegaHLibrary` must outlive all pcms objects;
  create it in `main` and `gc.collect()` before exit.

## 4. Phases

**Phase 0 (validation: reproduce theirs).** `pcms.Interpolator` point
evaluation at the 601x101 nodes (`--mode point` in section 5), diffed
against the bundled `example_traj` frames 0-9. This pins the channel order,
grid origin/extent and sampling method. Sanity check: the existing
checkpoint should behave the same on our frames.

**Phase 1 (P1 conservative).** Domain-matched P1 target; consistent for E/B,
lumped for density/velocity. Run side by side with Phase 0 output to
measure how much finer the mesh is than the grid near the cathode, and what
changes as a result.

**Phase 1b (reduced resolution).** P0 conservative frames at 600x100,
pooled to 300x50 and 150x25. Compare conservative vs point-sampled
quantities of interest at each resolution (section 6), then train at the
coarsest resolution that keeps them (section 7).

**Phase 2 (bounded + momentum).** Once the bounded projection is merged
(separate PR with its tests and bindings), switch
density/momentum to it, store momentum density, retrain or fine-tune.

## 5. Example code

### 5.1 Converter: `examples/python-api-example/mitl_exodus_to_frames.py`

Points marked `VERIFY` depend on the Exodus contents or on untested
behaviour.

```python
#!/usr/bin/env python3
"""Convert EMPIRE MITL Exodus runs into hnode plasma-gun frames with PCMS.

Writes <out>/<run>/frames/frame_%06d.pt (float32 [C, ny+1, nx+1]) and, with
--stats, <out>/stats.pt. Modes:
  point         pcms.Interpolator at grid nodes (reproduces the current data)
  conservative  conservative L2 projection onto P1 on the covered pixels
"""

import argparse
import gc
import os
import tempfile
from dataclasses import dataclass

import netCDF4
import numpy as np
import torch
from scipy.spatial import cKDTree

import pcms
import PyOmega_h as omega_h

# VERIFY: EMPIRE variable names and centering.
FIELD_CENTERING = {
    "Ex": "node", "Ey": "node", "Bz": "elem",
    "n_e": "elem", "vx": "elem", "vy": "elem",
}
# VERIFY: hnode channel order (inferred from the bundled frames).
VELOCITY_CHANNELS = ["Ex", "Ey", "Bz", "vx", "vy", "n_e"]
MOMENTUM_CHANNELS = ["Ex", "Ey", "Bz", "px", "py", "n_e"]

CONSISTENT = pcms.MassMatrixType.Consistent
LUMPED = pcms.MassMatrixType.Lumped
# Density-like channels must stay non-negative. Switch to the bounded
# projection (Farrell 2009) once it lands; lumped is the interim choice.
MASS_TYPE = {"Ex": CONSISTENT, "Ey": CONSISTENT, "Bz": CONSISTENT,
             "n_e": LUMPED, "px": LUMPED, "py": LUMPED}


def p_space(mesh, order):
    return pcms.LagrangeFunctionSpace.from_mesh(
        mesh, order, 1, pcms.CoordinateSystem.Cartesian,
        backend=pcms.LagrangeFunctionSpace.Backend.OmegaH)


def mesh_from_arrays(coords, tris, comm):
    """Build an Omega_h 2D triangle mesh via a gmsh 2.2 ASCII round trip.

    PyOmega_h has no build-from-arrays binding. Returns the mesh plus the
    permutations mapping input vertex/triangle order to Omega_h order
    (VERIFY whether the reader reorders; matching makes this irrelevant).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".msh", delete=False) as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(coords)}\n")
        for i, (x, y) in enumerate(coords, 1):
            f.write(f"{i} {x:.17g} {y:.17g} 0\n")
        f.write(f"$EndNodes\n$Elements\n{len(tris)}\n")
        for i, (a, b, c) in enumerate(tris, 1):
            f.write(f"{i} 2 2 1 1 {a + 1} {b + 1} {c + 1}\n")
        f.write("$EndElements\n")
        path = f.name
    try:
        mesh = omega_h.read_mesh_gmsh(path, comm)
    finally:
        os.unlink(path)

    oh_coords = mesh.coords().reshape(-1, 2)
    vert_perm = _match(coords, oh_coords)
    oh_tris = mesh.ask_elem_verts().reshape(-1, 3)
    elem_perm = _match(coords[tris].mean(axis=1),
                       oh_coords[oh_tris].mean(axis=1))
    return mesh, vert_perm, elem_perm


def _match(src_pts, oh_pts):
    """perm such that oh_values = src_values[perm]."""
    dist, idx = cKDTree(src_pts).query(oh_pts)
    if dist.max() > 1e-9 * np.ptp(src_pts):
        raise RuntimeError("mesh round trip changed entity locations")
    return idx


class ExodusRun:
    """netCDF4 view of one EMPIRE Exodus file (VERIFY layout)."""

    def __init__(self, path, comm):
        self.nc = netCDF4.Dataset(path)
        self.nc.set_auto_mask(False)
        coords = np.stack([self.nc["coordx"][:], self.nc["coordy"][:]], 1)
        nblk = len(self.nc.dimensions["num_el_blk"])
        self._blocks = range(1, nblk + 1)
        tris = np.concatenate(
            [self.nc[f"connect{b}"][:] - 1 for b in self._blocks])
        self.mesh, self._vperm, self._eperm = mesh_from_arrays(
            coords, tris, comm)
        self._elem_vars = _names(self.nc, "name_elem_var")
        self._nod_vars = _names(self.nc, "name_nod_var")

    @property
    def num_steps(self):
        return len(self.nc.dimensions["time_step"])

    def fields(self, step, names):
        out = {}
        for name in names:
            if name in self._elem_vars:
                i = self._elem_vars.index(name) + 1
                v = np.concatenate([self.nc[f"vals_elem_var{i}eb{b}"][step]
                                    for b in self._blocks])
                out[name] = np.ascontiguousarray(v[self._eperm], np.float64)
            else:
                i = self._nod_vars.index(name) + 1
                v = self.nc[f"vals_nod_var{i}"][step]
                out[name] = np.ascontiguousarray(v[self._vperm], np.float64)
        return out


def _names(nc, var):
    if var not in nc.variables:
        return []
    return [netCDF4.chartostring(row).strip() for row in nc[var][:]]


@dataclass(frozen=True)
class Grid:
    """nx x ny pixels over [x0, x1] x [y0, y1]; nodes are (ny+1, nx+1)."""

    x0: float
    y0: float
    x1: float
    y1: float
    nx: int = 600
    ny: int = 100

    @property
    def dx(self):
        return (self.x1 - self.x0) / self.nx

    @property
    def dy(self):
        return (self.y1 - self.y0) / self.ny

    def node_coords(self):
        X, Y = np.meshgrid(self.x0 + np.arange(self.nx + 1) * self.dx,
                           self.y0 + np.arange(self.ny + 1) * self.dy)
        return np.stack([X.ravel(), Y.ravel()], 1)

    @classmethod
    def from_mesh(cls, mesh, nx=600, ny=100):
        c = mesh.coords().reshape(-1, 2)
        return cls(*c.min(0), *c.max(0), nx, ny)


def pixel_coverage(src_mesh, grid, comm):
    """Covered-area fraction per pixel, shape (ny, nx): conservative P0
    projection of 1 onto a full-box pixel mesh."""
    box = omega_h.build_box(comm, omega_h.Family.SIMPLEX, 1.0, 1.0, 0.0,
                            grid.nx, grid.ny, 0, False)
    c = box.coords().reshape(-1, 2).copy()
    c[:, 0] = grid.x0 + c[:, 0] * (grid.x1 - grid.x0)
    c[:, 1] = grid.y0 + c[:, 1] * (grid.y1 - grid.y0)
    box.set_coords(np.ascontiguousarray(c.ravel()))
    src, tgt = p_space(src_mesh, 0), p_space(box, 0)
    op = pcms.OmegaHConservativeProjection(src, tgt, mass_matrix_type=LUMPED)
    fs, ft = src.create_field(), tgt.create_field()
    fs.set_dof_holder_data(np.ones(src_mesh.nelems()))
    op.apply(fs, ft)
    cent = ft.get_dof_holder_coordinates()
    ix = np.clip(((cent[:, 0] - grid.x0) / grid.dx).astype(int), 0,
                 grid.nx - 1)
    iy = np.clip(((cent[:, 1] - grid.y0) / grid.dy).astype(int), 0,
                 grid.ny - 1)
    area = np.asarray(box.ask_sizes())
    cov = np.zeros(grid.nx * grid.ny)
    np.add.at(cov, iy * grid.nx + ix, ft.get_dof_holder_data() * area)
    return cov.reshape(grid.ny, grid.nx) / (grid.dx * grid.dy)


class GridRasterizer:
    """Maps source-mesh DOF vectors to (ny+1, nx+1) node images.

    conservative: P1 target on the covered pixels only (domain-matched), one
    cached OmegaHConservativeProjection per (centering, mass type).
    point: P1 target on all grid nodes, pcms.Interpolator (Phase 0).
    """

    def __init__(self, src_mesh, grid, comm, mode, cov_tol=1e-12):
        self.grid, self.mode = grid, mode
        self._src = {"elem": p_space(src_mesh, 0), "node": p_space(src_mesh, 1)}
        self._src_field = {k: s.create_field() for k, s in self._src.items()}

        nodes = grid.node_coords()
        if mode == "conservative":
            cov = pixel_coverage(src_mesh, grid, comm)
            keep = np.argwhere(cov > cov_tol)            # (iy, ix) pixels
            n00 = keep[:, 0] * (grid.nx + 1) + keep[:, 1]
            n10, n01 = n00 + 1, n00 + grid.nx + 1
            n11 = n01 + 1
            tris = np.concatenate([np.stack([n00, n10, n11], 1),
                                   np.stack([n00, n11, n01], 1)])
        else:
            iy, ix = np.mgrid[0:grid.ny, 0:grid.nx]
            n00 = (iy * (grid.nx + 1) + ix).ravel()
            n10, n01 = n00 + 1, n00 + grid.nx + 1
            tris = np.concatenate([np.stack([n00, n10, n01 + 1], 1),
                                   np.stack([n00, n01 + 1, n01], 1)])
        used = np.unique(tris)
        local = np.full(len(nodes), -1)
        local[used] = np.arange(len(used))
        self._tgt_mesh, vperm, _ = mesh_from_arrays(
            nodes[used], local[tris], comm)
        self._node_index = used[vperm]       # Omega_h vertex -> grid node
        self._tgt = p_space(self._tgt_mesh, 1)
        self._tgt_field = self._tgt.create_field()
        self._ops = {}

        # Hat-function integrals w_i (lumped mass row sums) and mask.
        t = self._tgt_mesh.ask_elem_verts().reshape(-1, 3)
        a = np.asarray(self._tgt_mesh.ask_sizes())
        w = np.zeros(self._tgt_mesh.nverts())
        for k in range(3):
            np.add.at(w, t[:, k], a / 3.0)
        self.weights = self._scatter(w)
        self.mask = self._scatter(np.ones_like(w)) > 0
        self.coverage = (self.map(np.ones(src_mesh.nelems()), "elem", LUMPED)
                         if mode == "conservative" else self.mask * 1.0)

    def _op(self, centering, mass_type):
        key = (centering, mass_type)
        if key not in self._ops:
            src = self._src[centering]
            self._ops[key] = (
                pcms.OmegaHConservativeProjection(src, self._tgt,
                                                  mass_matrix_type=mass_type)
                if self.mode == "conservative"
                else pcms.Interpolator(src, self._tgt))
        return self._ops[key]

    def _scatter(self, v):
        img = np.zeros((self.grid.ny + 1) * (self.grid.nx + 1))
        img[self._node_index] = v
        return img.reshape(self.grid.ny + 1, self.grid.nx + 1)

    def map(self, values, centering, mass_type=CONSISTENT):
        f = self._src_field[centering]
        f.set_dof_holder_data(np.ascontiguousarray(values, np.float64))
        self._op(centering, mass_type).apply(f, self._tgt_field)
        return self._scatter(self._tgt_field.get_dof_holder_data())


def make_frame(rast, f, channels):
    """One [C, ny+1, nx+1] frame from the raw Exodus fields at a step."""
    def R(name, values, centering):
        img = rast.map(values, centering, MASS_TYPE.get(name, LUMPED))
        if rast.mode == "conservative":
            # Undo dilution on partially covered wall nodes (3.5).
            img = np.where(rast.mask, img / np.maximum(rast.coverage, 1e-300),
                           0.0)
        return img

    n = f["n_e"]
    out = {name: R(name, f[name], FIELD_CENTERING[name])
           for name in ("Ex", "Ey", "Bz", "n_e")}
    px = R("px", n * f["vx"], "elem")
    py = R("py", n * f["vy"], "elem")
    if "px" in channels:
        out["px"], out["py"] = px, py
    else:
        # Checkpoint-compatible velocity: momentum is only approximate (3.6).
        safe_n = np.where(out["n_e"] > 0, out["n_e"], 1.0)
        out["vx"] = np.where(out["n_e"] > 0, px / safe_n, 0.0)
        out["vy"] = np.where(out["n_e"] > 0, py / safe_n, 0.0)
    return np.stack([out[c] for c in channels]).astype(np.float32)


class RunningStats:
    """Streaming per-channel mean/std over in-mask nodes (Chan/Welford)."""

    def __init__(self, nchan):
        self.n = 0
        self.mean = np.zeros(nchan)
        self.m2 = np.zeros(nchan)

    def update(self, frame, mask):
        x = frame[:, mask].astype(np.float64)
        nb = x.shape[1]
        mb, vb = x.mean(1), x.var(1) * nb
        d = mb - self.mean
        tot = self.n + nb
        self.mean += d * nb / tot
        self.m2 += vb + d ** 2 * self.n * nb / tot
        self.n = tot

    @property
    def std(self):
        return np.sqrt(self.m2 / self.n)


def convert_run(exo_path, out_dir, args, comm, stats=None):
    run = ExodusRun(exo_path, comm)
    grid = Grid.from_mesh(run.mesh, args.nx, args.ny)   # VERIFY vs Phase 0
    rast = GridRasterizer(run.mesh, grid, comm, args.mode)
    channels = (MOMENTUM_CHANNELS if args.momentum else VELOCITY_CHANNELS)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for step in range(0, run.num_steps, args.stride):
        frame = make_frame(rast, run.fields(step, FIELD_CENTERING), channels)
        torch.save(torch.from_numpy(frame),
                   os.path.join(frames_dir,
                                f"frame_{step // args.stride:06d}.pt"))
        if stats is not None:
            stats.update(frame, rast.mask)
    return rast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="name=path.exo pairs, one per (Vpeak, seed)")
    p.add_argument("--train", nargs="*", default=None,
                   help="run names that contribute to stats.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["point", "conservative"],
                   default="conservative")
    p.add_argument("--momentum", action="store_true",
                   help="store n*v instead of v (exactly conservative)")
    p.add_argument("--nx", type=int, default=600)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    lib = omega_h.OmegaHLibrary()      # must outlive every pcms object
    comm = lib.world()

    def run_all():
        nchan = len(MOMENTUM_CHANNELS if args.momentum else VELOCITY_CHANNELS)
        stats = RunningStats(nchan)
        rast = None
        for spec in args.runs:
            name, path = spec.split("=", 1)
            use = args.train is None or name in args.train
            rast = convert_run(path, os.path.join(args.out, name), args,
                               comm, stats if use else None)
        torch.save({
            "mask": torch.from_numpy(rast.mask.astype(np.float32)),
            "mean": torch.from_numpy(stats.mean.astype(np.float32)),
            "std": torch.from_numpy(stats.std.astype(np.float32)),
            "weights": torch.from_numpy(rast.weights),
            "coverage": torch.from_numpy(rast.coverage),
        }, os.path.join(args.out, "stats.pt"))

    run_all()
    gc.collect()


if __name__ == "__main__":
    main()
```

Notes on the sketch:
- `stats.pt` takes the mask from the last run converted. This assumes all
  runs share one mesh; assert that across runs when the data arrives.
- One process per run is the intended parallelism. Writing stats then needs
  a merge step (Chan's parallel formula, as in `RunningStats.update`) rather
  than the single loop shown.
- `--mode point` builds the target on all grid nodes, including nodes
  outside the domain. Their values depend on `OutOfBoundsPolicy` and are
  masked anyway; compare them against the bundled frames only inside the
  mask.
- In point mode the mask is "node inside the target mesh", which covers the
  full box. For the Phase 0 diff use their `stats["mask"]`; for training use
  our domain-matched mask from conservative mode.

### 5.2 hnode side

The existing `Horseshoe2DDataset` loads the converter's output unchanged:
same file layout, the same `[C, 101, 601]` shape, and `stats.pt` keys
`mask` / `mean` / `std` (extra keys are ignored). Only the run list changes:

```python
DIRECTORIES = [f"V{v}_s{s}" for v in ("0.9", "1.0", "1.1") for s in range(4)]
DIRECTORY_Z_MAP = torch.tensor([z for z in (-1.0, 0.0, 1.0) for _ in range(4)])
```

Physical diagnostics, for validation and for checking surrogate rollouts.
The weights make these exact for the conservative frames:

```python
def electron_diagnostics(frame, stats, n_ch=5, x0=0.0, dx=50.0 / 600):
    """Total electron count and x centre of mass from one un-normalized
    frame [C, H, W]. Uses the P1 hat-function weights from stats.pt, so the
    count equals the EMPIRE mesh integral for conservative frames."""
    w = stats["weights"].to(frame.dtype)
    n = frame[n_ch] * stats["mask"].to(frame.dtype)
    x = x0 + dx * torch.arange(frame.shape[-1], dtype=frame.dtype)
    total = (w * n).sum()
    x_cm = (w * n * x).sum() / total
    return total, x_cm
```

The paper's quantity of interest is the electron centre of mass *before* the
expanded region; restrict the sums to columns `< 420` for that.

Optional, for retraining: replace the binary mask channel with coverage.
This is a one-line change in `Horseshoe2DDataset.__init__`; the loss mask
stays binary:

```python
self.mask = stats["mask"].double().unsqueeze(0).unsqueeze(0)
self.input_mask = stats.get("coverage", stats["mask"]).double()[None, None]
# __getitem__: torch.cat([data, self.input_mask.repeat(...)], dim=1)
```

## 6. Reduced-resolution training data

### 6.1 Why coarsening is where conservation pays off

At 600x100 the grid is roughly at mesh resolution. The domain is about 58%
of a 50 x 8.3 mm box (~240 mm^2, assuming square pixels), so 55,065
triangles give **~1.6 triangles per pixel** on average. That is why the gain
at native resolution is expected to be small. Coarsening changes this:

| Grid | Triangles per pixel (avg) | Point sampling | Conservative |
|---|---|---|---|
| 600x100 | ~1.6 | Mild aliasing | Exact integrals |
| 300x50 | ~6.4 | Keeps ~1 in 6 cells; the rest are ignored | Exact integrals, proper averaging |
| 150x25 | ~26 | Mostly aliasing and noise | Exact integrals |

In an earlier PCMS ML-ingestion study on analytic fields (scripts not in
the repository), point-sampling error stalled near 1e-2 regardless of
resolution, while conservative error dropped to 5e-6. At coarse resolution, averaging also reduces PIC shot noise, which
point sampling passes through unchanged.

The local grading of the EMPIRE mesh is unknown; the ratio of cell size to
pixel size in the plasma region (cathode, plasma ball, charge islands)
decides how much point sampling loses at each resolution.

### 6.2 Generating coarse datasets

With P0 on nested grids, the conservative cell average on a coarse grid is
exactly the mean of the fine cell averages it contains. So:

1. Run PCMS once at 600x100 (P0, lumped).
2. Produce 300x50 and 150x25 by 2x2 and 4x4 average pooling of the frames
   (and of coverage). Exact and nearly free; no reprocessing of the 1.1 TB.

`stats.pt` (mean/std, mask, coverage) must be recomputed per resolution.
For the mask, use coarse coverage > 0.

### 6.3 Caveats

- Coarsening smears the thin emission sheath at the cathode and the small
  charge islands in Fig. 3b.
- The channel's quantity of interest holds up better: at 300x50 the gap is
  still ~20 pixels across, the total electron count stays exact, and the
  centre of mass across the gap stays accurate to well under a pixel.
- Any architecture change (section 7.2) means training from scratch.

## 7. Training notes

### 7.1 Reported cost and where the loss levels off

- Paper Table 1: Inner MITL, **12 days** of training (batch 8, LR 1e-4, L1).
  The single-H200 hardware is stated explicitly only for shear flow (8 days);
  presumably the same here.
- The notebook uses `batch_size=1`, which conflicts with Table 1's batch 8.
  Per-step cost depends on which was actually run.
- The released checkpoint (`swin_plasma_e2e_redo_part_2.pth`) stores 144,900
  optimizer steps of loss history, so about **7.2 s/step** if the 12 days
  refers to this run.

Smoothed (500-step) normalized L1, relative to the final value:

| Relative to final | Reached at step | Share of run | Wall time at 7.2 s/step |
|---|---|---|---|
| 1.5x | ~7.3k | 5% | ~15 h |
| 1.2x | ~21k | 15% | ~1.8 days |
| 1.1x | ~42k | 29% | ~3.5 days |
| 1.05x | ~69k | 47% | ~5.7 days |

Most of the 12 days bought the last 10%. A run to ~1.1x of their final loss
is a few days, not 12.

### 7.2 Input size and token count

Their Swin (`swin_transformer_v2.py:72`) uses one window size (it can be
rectangular) across all four stages, after patch-4 embedding and three
2x downsamplings. So the padded input height must be a multiple of 32*Wh and
the width a multiple of 32*Ww, where (Wh, Ww) is the window size. With the
current (8,8) window, 256x768 is the smallest size that fits 101x601:
**69% of the input is padding.**

| Input | Padded to | Window | Tokens vs current |
|---|---|---|---|
| 101x601 (current) | 256x768 | (8,8) | 1.0 |
| 101x601 | 128x768 | (4,8) | 0.5 |
| 51x301 or 50x300 | 64x384 | (2,4) | 0.125 |

With windowed attention, compute and activation memory scale roughly
linearly with token count.

- This assumes the Swin encoder/decoder dominates step cost. The latent
  HNODE is 64-dimensional, so this is plausible, but it is not profiled.
  Fixed costs (latent Newton solves, I/O, kernel launches) will not shrink
  with the input, so 8x fewer tokens will give less than 8x.
- Tighter padding (the (4,8) row) needs no PCMS. The honest pitch is "trim
  the padding, then coarsen conservatively"; only the second half is PCMS.
- Changing the window or input size changes the relative-position bias
  tables, so the existing checkpoint will not load: train from scratch.
- At 150x25 the window height would have to be 1, which removes vertical
  mixing within windows. That resolution needs a different patch size or
  fewer downsampling stages.
- Fewer pixels also cut disk reads. Frames are float32 6x101x601 (~1.46 MB),
  so ~17.5 GB per run and ~210 GB for all 12 runs at native resolution (the
  paper's 1.1 TB must include raw or extra data). At 300x50 that is ~53 GB,
  small enough for node-local NVMe or RAM. Each sample loads 11 frames with
  `torch.load`, so I/O may matter at native resolution.

### 7.3 Distributing training

Data-parallel training should work with modest changes:

- **Gradients take the ordinary autograd path.** Newton iterations in
  `_timestep` (`hybrid_node_solver.py:664`) run under `no_grad`; a final
  Jacobian/residual evaluation and one linear solve run with autograd on.
  `loss.backward()` fills `.grad` normally, with no hand-written parameter
  gradients.
- **The model is small** (72M parameters): no model sharding needed.
- **Samples are independent** (one random window from one run each), so
  ranks draw their own with `DistributedSampler`. Dropout is 0, so no
  shared-seed issues.
- **SOAP is deterministic given the gradients**: each rank keeps an
  identical optimizer copy; its preconditioner update every 10 steps is
  duplicated per rank (wasted compute, but correct).

`nn_rhs_model` is called many times per step (every Newton iteration of
each of the 32 rollouts, mostly under `no_grad`), and
`DistributedDataParallel` is fragile with repeated forwards before one
backward. Skip the wrapper and average gradients by hand:

```python
import torch.distributed as dist

params = [p for m in (encoder, decoder, nn_rhs_model) for p in m.parameters()]
# after loss.backward(), before optimizer.step():
for p in params:
    if p.grad is not None:
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
```

Plus: identical initial weights on all ranks (broadcast from rank 0 or load
the same checkpoint) and a per-rank sampler. About 30 lines in the training
cell, launched with `torchrun`. The exchange is ~580 MB per step (72M
float64 parameters): negligible against ~7 s per step on NVLink, fine over
InfiniBand.

What it buys:
- **A bigger batch, not a faster step.** 8 GPUs give batch 8 at roughly the
  time of one batch-1 step. Fewer steps to a given loss (less gradient
  noise), but not 8x fewer; the learning rate may need retuning.
- **Uneven load.** Samples converge in different numbers of Newton
  iterations, so ranks wait at each gradient exchange.

### 7.4 Precision

The whole model trains in float64 (`torch.set_default_dtype(torch.float64)`),
including the 72M-parameter Swin encoder/decoder. On an H200, float64
throughput is a small fraction of TF32/BF16. Keeping the latent Newton solve
in float64 while running the encoder/decoder in float32 or BF16 could
plausibly cut step time several-fold, on top of data parallelism.

This changes their method. Their stability argument concerns the latent
integrator, which stays float64, but treat it as an experiment. For
comparisons, run every arm with identical settings.

### 7.5 Running on B200 nodes billed per node-hour

- **Fill the node with independent runs rather than distributing one.** A
  single-GPU job pays for 7 idle GPUs. Eight single-GPU runs cost the same
  node-hours, need no distribution code or LR retuning, and give every
  comparison arm at once. Example layout:

  | GPU | Run |
  |---|---|
  | 0-1 | 300x50 from scratch: conservative vs point-sampled |
  | 2-3 | Same, second seed (separates the effect from seed noise) |
  | 4-5 | Same comparison in mixed precision, if the benchmark shows it trains stably |
  | 6 | Fine-tune the existing checkpoint on PCMS frames at 601x101 |
  | 7 | Spare: streaming demo (section 8), or a native 128x768 baseline |

  At ~3.5 days to ~1.1x their final loss, about 84 node-hours for
  everything, before any precision or resolution speedup.
- **Float64 is a worse deal on a B200.** Approximate spec-sheet numbers (to
  verify): B200 FP64 peak ~40 TFLOPS, below an H200's FP64 tensor-core peak
  of ~67, while its BF16/TF32 throughput is far higher. The all-float64 model
  is probably no faster per step than on their hardware. This makes the
  precision benchmark the most valuable week-1 item.
- **Keep non-GPU work off the node.** Frame conversion runs on a CPU
  partition. Stage frames on node-local NVMe.
- **Stop at the plateau, not at 12 days.** Checkpoints save every 100 steps.
- **Cost uncertainty:** if their MITL run was 8 GPUs data-parallel at batch 1
  each (matching Table 1's batch 8), the 12 days were node-days and
  single-GPU runs will be slower than 7.2 s/step. The benchmark settles this.

**A first pass does not need all 12 runs.** One run gives ~11.5k training
windows (their 144,900 steps over 12 runs is about one pass through the
data). Train on one run at 1.0 MV (z = 0, same setting as the bundled
`example_traj`) with constant z, and evaluate on a held-out seed at the same
voltage. Add one seed per voltage only if the talk claims voltage
interpolation. GPU cost is set by step count, not dataset size; fewer runs
save transfer and conversion. Keep every stored frame for training (frame
offsets assume dt = 1e-2 per frame); striding is fine for diagnostics only.

### 7.6 Schedule for a talk in a few weeks

**Week 1: data and loader. No GPU needed; guaranteed talk content.**
- Get the Exodus files; write the reader for EMPIRE's layout.
- Phase 0: reproduce their frames.
- Per-step electron count and centre of mass: their point-sampled frames vs
  PCMS conservative frames vs EMPIRE, at 600x100, 300x50 and 150x25 (pooled
  P0). This is a talk figure even if no training finishes.
- Benchmark ~100 training steps each: batch 1 vs 8 ranks, float64 vs mixed
  precision, native vs 64x384 input. Replaces the estimates above with
  measured numbers.

**Week 2: training, chosen by GPU budget.**
- *Cheapest:* fine-tune the existing checkpoint on PCMS P1 frames at
  601x101 with velocity channels. Unchanged architecture, so the checkpoint
  and optimizer state load directly; 1-2 days. Shows the frames work as a
  drop-in, but with no comparison.
- *Better story:* from scratch at 300x50 (padded to 64x384, window (2,4)),
  twice with the same step budget: conservative frames vs point-sampled
  frames. Compare electron count and centre of mass in surrogate rollouts.
  Two runs of a few days, in parallel or back to back.

**Week 3:** buffer, long rollouts, figures.

### 7.7 Risks, most likely first

1. **Moving the data.** The processed dataset alone was 1.1 TB; raw Exodus
   is probably larger. Find where it lives and whether conversion can run
   next to it. Conversion itself parallelizes trivially (by run and by
   time-step chunk) and will not be the bottleneck.
2. **GPU access.** Ideally H200/H100-class, two for the comparison runs.
3. **EMPIRE's file layout.** Per-rank output means extra reader work.
4. **The coarse architecture.** From-scratch training at a new size may
   need LR tuning; keep the fine-tune path as the fallback.

## 8. Streaming demo (ADIOS2)

An alternative or complement to the training study: stream native mesh
fields from a simulation (Exodus replay standing in for EMPIRE) over ADIOS2,
ingest them conservatively on the ML side, and run the **pretrained**
surrogate in the loop. No training needed.

### 8.1 What exists

- **The coupler already streams over ADIOS2 in C++.** It runs on redev,
  which opens ADIOS2 channels with BP4 (file-based) or SST (live stream)
  transport (`redev_adios_channel.h`). `ApplicationComm` provides the
  per-step API: `AddField` / `BeginSendPhase` / `SendField` / `ReceiveField`
  (`coupler.hpp`). Fields sharing a layout share one communicator, so the 6
  cell fields ride one exchange plan.
- **The communication layer already handles partitioned meshes.** The
  XGC/proxy coupling tests (`test/test_proxy_coupling*.cpp`, registered with
  `dual_mpi_test` / `tri_mpi_test` in `test/CMakeLists.txt`) run clients on
  pre-partitioned `.osh` meshes such as `d3d-full_9k_sfc_p8.osh`. The server
  partition can be `redev::ClassPtn` (model classification + `.cpn` file) or
  `redev::RCBPtn` (coordinates); both are dispatched in
  `src/pcms/coupler/partition.h`.
- **The transfer operators are serial-only.** Conservative projection builds
  its PETSc system from a per-rank GID-sorted numbering, which is not a valid
  global row numbering on several ranks. Any rank that runs the rasterizer
  must hold the whole mesh.
- **The coupler is bound in Python** (`bind_coupler.cpp`, 8.4): `CouplerComm`,
  `ApplicationComm`, field and data handles, BP4/SST, ADIOS2 parameters,
  mpi4py communicators. Before 2026-10-27 `src/pcms/pythonapi` covered
  spaces, fields and transfer operators only.
- **Single-process prototype: `examples/mitl-streaming/`.** See its
  `README.md`. Working: analytic synthetic EMPIRE stand-in on a ~55k-triangle
  mesh (`synthetic_source.py`), conservative/point rasterizer with exact
  electron count (`rasterizer.py`), in-process SST-like queue with
  block/discard policies (`LoopbackTransport`), consumer with re-anchoring
  and gap handling (`consumer.py`), cheap models behind the `rollout`
  interface (`models.py`), `unittest` suites. `PcmsTransport` in
  `transport.py` runs the same `send` / `close` / `messages` interface over
  the coupler between two programs (8.9).
- **Development environment:** with ADIOS2 built with SST and MPI, plus
  mpi4py (section 0), the bindings and a two-process SST test can be
  developed on a workstation or laptop; no cluster or GPU is needed until
  S5-S7.

#### EMPIRE test data

One EMPIRE MITL run is in the PCMS test data repo
(https://github.com/SCOREC/pcms_testcases, `PCMS_TEST_DATA_DIR`):
`exodus/flow/flow.exo` (293 MB, Git LFS; `step-4.png` is a rendering of it).
Contents, read from the file header on 2026-10-27:

- **Format:** Exodus II, netCDF classic 64-bit offset (CDF-2), float64.
  Readable with `netCDF4` or `scipy.io.netcdf_file`; no SEACAS needed.
- **Mesh:** 2D, 28,117 nodes, 55,065 triangles in one block (`block_1`),
  the same mesh as the paper. Coordinates in metres. The domain is an exact
  L: a 50 x 8.25 mm box minus x < 35 mm, y < 5 mm (area 237.5 mm^2, all
  boundary edges axis-aligned). x = 35 mm is a grid line of the 600 x 100
  frame grid, y = 5 mm is not (row 60.6); it is for ny a multiple of 33, so
  600 x 99 (square 1/12 mm pixels) conforms exactly. 13 side sets (`anode`, `cathode`, `input`,
  `output`, `load`, `mitl_anode`, `mitl_cathode`, emitter/non-emitter
  surfaces, ...), 3 node sets.
- **Time:** 51 steps, 0 to 10 ns every 0.2 ns. The paper's training data has
  11,991 snapshots per run (about 0.83 ps apart), so this file is a coarse
  subset of one run; which voltage/seed is unknown.
- **Fields (all cell-centred, 13):** `LOAD_BAL` (all zero), `PROC_ID`,
  and electron moments: `e- Number Density`, `e- Count`,
  `e- Count Density`, `e- Average VelocityX/Y/Z`,
  `e- Current DensityX/Y/Z`, `e- Average Kinetic Energy`, `e- Energy`.
  At the last step 43,209 of 55,065 cells hold electrons; peak number
  density ~2.8e18.
- **No electromagnetic fields.** Ex, Ey, Bz are not in the file.
- **Native decomposition:** the file was joined from per-rank output by
  `epu`, and `PROC_ID` records EMPIRE's 8-rank decomposition (static, about
  6,883 cells per rank). Element index in the joined file is the global ID.

Consequences for the demo:

- Step 2 can use EMPIRE's own decomposition (8.5); no offline partitioner.
- **The pretrained hnode checkpoint cannot run on this file as-is.** It
  expects 6 channels including Ex, Ey, Bz, at a ~240x finer time spacing.
  The ingestion side (stream, project, exact electron count) is unaffected.
  For the surrogate-in-the-loop part, either get a full-cadence output with
  fields from EMPIRE, or show the in-loop model on the electron channels
  only with a cheap model (`models.py`), stated as such. Decide before S5.

### 8.2 Architecture

```
 EMPIRE replay (producer, N ranks)  ADIOS2 SST        Consumer (1 rank, GPU)
 ──────────────────────────  ──────────────────▶  ─────────────────────────────────
 ExodusRun: step k fields                          PCMS ReceiveField x6 (full mesh)
 PCMS ApplicationComm                              GridRasterizer -> frame (exact count)
   BeginSendPhase                                  ring buffer (last 3+ frames)
   SendField x6 (owned part)                       encoder -> latent -> HNODE rollout
   EndSendPhase                                    decoder -> forecast frames
                                                   view: truth | forecast | error,
                                                   electron count trace, re-anchor every K
```

- **Projection happens on the consumer.** The simulation sends native mesh
  data; the grid view is the ML side's choice. The rasterizer and cached
  operators from section 3 are reused unchanged.
- **Pretrained checkpoint, inference only.** Phase 0 interpolation frames
  match its training data; conservative P1 frames shown alongside.
- **Re-anchor every K steps** by re-encoding the latest streamed frames;
  shows surrogate drift against live truth.
- **Queue policy: start with `QueueFullPolicy=Block`.** The goal is that ML
  never throttles the simulation (`Discard` with a small `QueueLimit`), but
  redev exchanges its setup messages over the same channel, and discarding
  those would break the handshake. Verify before switching (task S2).
- **End of stream:** a small `meta` array `[step, time]` is sent with every
  step through `AddData`; `step = -1` is the end-of-stream sentinel.
- **Bandwidth is trivial:** ~55k cells x 6 fields x 8 B ~ 2.6 MB per step.
- Both participants read the mesh from disk (no mesh-over-the-wire today).
- Step 1 runs the producer on one rank; step 2 (section 8.5) runs it on N.

### 8.3 What the demo shows, and what it does not

Shows:

- Live simulation data ingested into an ML model through PCMS, from
  Python, over the same ADIOS2 transport the production couplings use.
- Exact conservation at ingestion: the per-frame electron count matches the
  mesh integral (~1e-14) at every step, next to the point-sampled pipeline,
  which does not.
- A surrogate running against streamed truth, with drift visible between
  re-anchors.
- With step 2: redistribution from a decomposed simulation to the ML
  process, which a plain ADIOS2 stream would not do.

Tests (engineering):

- The Python coupler bindings, including exact round trip of P0/P1 fields
  over BP4 and SST.
- SST on the target cluster.
- The hnode checkpoint running inside the PCMS pipeline.

Does not show:

- That conservative ingestion improves the surrogate. That is the training
  study (section 7). The checkpoint was trained on point-sampled frames, so
  conservative frames are slightly out of distribution for it; show them
  alongside, do not swap them in.
- Live EMPIRE: it is an Exodus replay (or the synthetic stand-in).
- Scale or performance.
- Without step 2, much of the coupler: a single-rank producer and a
  single-rank server never exercise the redistribution.

### 8.4 Proposed Python coupler API

Implemented (`src/pcms/pythonapi/bind_coupler.cpp`); the proposal changed
where noted below.

```python
import pcms

server = role == "consumer"
partition = pcms.RCBPartition(2, [0], [0.0]) if server else None  # one leaf
coupler = pcms.CouplerComm("mitl", comm, server, partition)  # comm: mpi4py
                                                            # or None (world)
app = coupler.add_application(
    "empire", path=".", transport=pcms.TransportType.SST,       # or BP4
    params={"OpenTimeoutSecs": "300", "DataTransport": "WAN",
            "QueueLimit": "4", "QueueFullPolicy": "Block"})

space = pcms.LagrangeFunctionSpace.from_mesh(
    mesh, 0, backend=pcms.LagrangeFunctionSpace.Backend.OmegaH,
    layout_name="elem")                    # layout_name: required, both sides
handles = {name: app.add_field(name, space) for name in names}
meta = np.zeros(2)                                             # [step, time]
meta_handle = app.add_data("meta", meta)

# producer, per step
app.begin_send_phase()
for h in handles.values(): h.send()
meta_handle.send()
app.end_send_phase()

# consumer, per step: begin_receive_phase / receive() / end_receive_phase,
# then h.field.get_dof_holder_data()
```

- MPI communicator from mpi4py: `comm.py2f()` -> `MPI_Comm_f2c` in the
  binding; avoids a compile-time dependency on mpi4py headers.
- `add_field(name, space)` creates the field inside the application (C++
  `AddField` takes it by move), so Python never holds a moved-from field;
  `handle.field` is the registered one. Handles keep the application alive,
  the application keeps the coupler alive, and the data array stays alive
  while registered (pybind11 `keep_alive`).
- `RCBPartition(dim, ranks, cuts)` is exposed instead of a
  `single_rank_partition()` helper; one server rank is `RCBPartition(2, [0],
  [0.0])`.
- `from_mesh` gained `layout_name` (the coupler requires named layouts) and
  `owned_mask` (8.5).

### 8.5 Step 2: partitioned producer (N -> 1)

Run the producer on N ranks, each owning part of the EMPIRE mesh and sending
only its owned DOFs; the consumer stays on one rank and receives the
assembled field by global ID. This is how a live EMPIRE coupling would look
(EMPIRE runs decomposed) and it exercises the exchange planner, the part of
PCMS a raw ADIOS2 stream does not replace. The data is small, so this is
about the mechanism, not need.

**Why N -> 1 and not N -> M.** The consumer runs the conservative transfer,
which is serial-only (8.1). N -> M needs partitioned transfer operators with
a global PETSc row numbering, a separate roadmap item. Out of scope.

**Producer partition.**

- Use EMPIRE's native 8-rank decomposition from the `PROC_ID` cell variable
  (8.1, EMPIRE test data). Each producer rank reads the joined file with the
  serial reader, keeps the cells with its `PROC_ID`, and builds its local
  mesh piece with global IDs = element index in the file; nodes on rank
  boundaries are shared, so node GIDs = node index and ownership must be
  assigned (e.g. lowest rank). Run the producer with 8 ranks (or merge
  `PROC_ID`s for fewer).
- For the synthetic source, or a different rank count, partition offline
  into a pre-partitioned `.osh` (the same form as the `_p8.osh` test
  meshes).
- Required invariant: the partitioned mesh's global IDs match the entity
  numbering of the serial mesh the consumer reads. Test it explicitly (P2).

**Server partition.** With one server rank the partition is trivial. If a
real one is ever needed, prefer `RCBPtn`: `ClassPtn` needs model
classification and a `.cpn` file, and a mesh that came through the gmsh
round trip likely has only trivial classification.

**Resolved: a distributed Omega_h mesh from Python.** Rank 0 writes the
full mesh with `write_mesh_binary`, all producer ranks call
`read_mesh_binary(path, world)` (a one-part file read on N ranks puts every
element on rank 0) and then `mesh.balance()`, which repartitions it by
recursive inertial bisection. Omega_h keeps the file order as the global
numbering, so the "global" tag is the Exodus index the consumer uses, and
Omega_h's own vertex ownership is what the coupler needs: `from_mesh` takes
no mask. This is the producer's default
(`examples/mitl-streaming/partition.py`, `balanced_mesh`). Reading the gmsh
file on N ranks, or on rank 0 followed by `set_comm`, segfaults; only the
binary route works.

The partition is Omega_h's, not EMPIRE's `PROC_ID`. Imposing a given
partition on the distributed mesh needs `Mesh::migrate`, which PyOmega_h does
not bind. For that case `--partition source` keeps the first approach: each
producer rank builds a *serial* mesh of its own piece on a single-rank
communicator (`lib.world().split(rank, 0)`), sets the "global" tag of
elements and vertices to the full-mesh index with `set_tag`, and passes a
vertex `owned_mask` (lowest touching rank owns) to `from_mesh`. The coupler
only needs owned flags, GIDs, coordinates and classification per DOF holder,
so pieces on separate communicators work too.

### 8.6 Tasks

| # | Task | Estimate | Notes |
|---|---|---|---|
| S1 | Python bindings: `CouplerComm`, `ApplicationComm`, `add_field` / `add_data`, send/receive phases, transport enum, ADIOS2 params dict, mpi4py communicator (8.4) | 2-4 days | **Critical path.** Own PR with tests. Single-rank server. |
| S2 | Two-process ctest (`dual_mpi_test`): P0/P1 field over BP4, then SST, exact arrival; check whether `Discard` survives the redev handshake | 0.5-1 day | Locks in S1 before any ML |
| S3 | Synthetic producer | done | `examples/mitl-streaming/synthetic_source.py` (analytic run on a jittered mesh) |
| S3b | Implement `PcmsTransport` against S1; run the prototype as two processes | 0.5 day | Stub and intended calls already in `transport.py` |
| P1 | Partitioned meshes from Python: check PyOmega_h, add a read helper if needed; split `flow.exo` by `PROC_ID` (offline partition for the synthetic mesh) | 0.5-2 days | Size depends on the PyOmega_h check (8.5) |
| P2 | N -> 1 ctests (2 -> 1 and 4 -> 1): owned-DOF send, exact assembled field on the server, GID invariant | 1 day | Reuse `dual_mpi_test` |
| P3 | Partitioned producer in the demo (`run_producer` on N ranks) | 0.5-1 day | Consumer unchanged |
| S4 | Real producer: `ExodusRun` replay of `exodus/flow/flow.exo` at a configurable rate | 0.5 day | Reuses section 3.1; data is in the test data repo (8.1) |
| S5 | Consumer: hnode checkpoint adapter behind the `rollout` interface | 1-2 days | Rasterize/buffer/re-anchor already in `consumer.py`; inference code from the notebook; GPU for inference only |
| S6 | Recorded run (frames/video), then live view | 1 day | Recording is the talk-day fallback |
| S7 | SST on the cluster | 0.5-2 days | Biggest unknown. Reduced scope: same node, sockets |

Total about 2-3 weeks of development including step 2, a few node-hours.
Everything up to S5's inference can be developed and tested off the GPU node.

**Reduced scope (default):** producer and consumer on one GPU node, SST over
sockets, BP4 as fallback through the same code path, pretrained checkpoint,
recorded run first. Out of scope: cross-node RDMA, live EMPIRE (would need
EMPIRE to link PCMS), in-situ training, multi-rank consumer.

**Optional bridges to training:** online fine-tuning on streamed windows
(shown working, not converged); the consumer writing frames to disk builds
the training dataset as a by-product.

### 8.7 Recommended sequence

1. Step 1, now (no data needed): S1 bindings, S2 test, S3b two-process
   prototype over the PCMS transport.
2. Step 2: P1 (PyOmega_h check first), P2, P3.
3. When data arrives: shared foundation + conservation diagnostics
   (section 4, Phase 0/1); S4 replay; training setup on GPUs 0-6 if doing
   training.
4. S5-S7 on the GPU node (GPU 7 while training runs).
5. Talk: conservation-diagnostics figure + streaming demo; training as
   follow-up, or a result if it finished.

### 8.9 Status (2026-10-27)

| # | State | Where |
|---|---|---|
| S1 | Done | `src/pcms/pythonapi/bind_coupler.cpp`; `from_mesh(layout_name=, owned_mask=)` |
| S2 | Done | ctests `python_coupler_exchange_{BP4,SST}` (P0+P1+data, both directions, bit for bit) |
| S3b | Done | `PcmsTransport`, `run_producer.py` / `run_consumer.py` / `run_pcms_demo.sh` |
| P1 | Done | `partition.py`: Omega_h-balanced distributed mesh (default), or serial pieces split by `PROC_ID` (8.5) |
| P2 | Done | ctests `python_coupler_partitioned_{pieces,balanced}_{2,4}to1_{BP4,SST}` (values encode GIDs); SST needs `SpeculativePreloadMode=OFF`, see below |
| P3 | Done | `run_pcms_demo.sh 8 --source exodus ...` |
| S4 | Done | `exodus_source.py` (netCDF4; mm units; electron moments) |
| S5 | Adapter done | `hnode_model.py`, `hnode_smoke.py`; CPU only here |
| S6 | Recorded runs | `--plot` / `--record` of the runs below; no live view |
| S7 | Not started | needs the cluster |

Measured on a workstation container (8 cores, no GPU):

- `flow.exo`, all 51 steps, 8 producer ranks -> 1 consumer over SST:
  every received field equal to the file bit for bit (`--verify`),
  electron count of the conservative 600 x 99 frames equal to the mesh
  integral to 2e-15 at every step.
- The same with 1 and 2 producer ranks, and over BP4.
- The same, 8 ranks, on the Omega_h-balanced distributed mesh (now the
  default; 6880-6885 elements per rank): bit for bit and 1.7e-15 again,
  ~300 ms/step, 3 of 3 runs over SST and 1 over BP4.
- Synthetic source (6 channels, 600 x 100) in one process (loopback): count
  exact to 6e-15, ~1.2 s/step for the 6 projections (CPU).
- hnode checkpoint on its bundled frames 0-2 -> 3-9 through `rollout`:
  normalized masked L1 2.7e-3 (its training loss ends near 7e-2), 11 s for
  7 frames on CPU; it also runs inside the two-program consumer on the
  synthetic source (forecasts meaningless there: not its data).

Findings:

- **SST `QueueFullPolicy=Discard` does not work with redev** (open question
  12). With `QueueLimit=1` the channel setup fails (client times out finding
  the server's writer). With `QueueLimit=4` setup succeeds, but steps
  including the end-of-stream marker can be dropped; the consumer then gets
  ADIOS2 EndOfStream and redev aborts (`redev_adios_channel.h:158` asserts
  `StepStatus::OK`). Making ML never throttle the simulation needs redev to
  keep its setup out of the discardable queue and to report end of stream,
  or a separate channel. The demo uses Block.
- **N -> 1 over SST crashed intermittently: an ADIOS2 bug, worked around
  with `SpeculativePreloadMode=OFF`.** With several client ranks the server
  segfaulted during coupler setup (`add_field`): 6/20 runs at 4 clients with
  serial pieces, 10/20 with the balanced mesh; never 1 -> 1 or over BP4.
  With a debug, AddressSanitizer build of ADIOS2 2.12.1 the fault is a NULL
  read in `HandleRequestWithPreloaded` (`toolkit/sst/dp/evpath_dp.c`), on
  the reader's network thread. A reader with few ranks gets speculative
  preload by default (`SpeculativePreloadMode=AUTO`): every writer rank
  pushes its data for each step to the reader. When a preload arrives,
  `EvpathPreloadHandler` matches every pending read request against the
  *preload's* timestep instead of the request's (the request handle does not
  even record its timestep). redev's first setup step is written by client
  rank 0 only, so the other ranks' preloads for it are empty, and each
  writer rank's data connection opens on its first preload, so those empty
  preloads can arrive after the reader has advanced and requested the next
  step's block from the same rank. The request is then "satisfied" from the
  empty entry: memcpy from NULL. With data in the stale entry it would
  silently copy the wrong step's values instead. SST's verbose trace of a
  failing run shows exactly this order. Still present on ADIOS2 master.
  Not a PCMS or redev bug; redev's deferred-`Get` buffers are fine.

  Evidence: a standalone ADIOS2 reproducer (no PCMS/redev; 8 writers, step
  0 written by rank 0 only) crashes the reader in 17/40 runs, 0/40 with
  `SpeculativePreloadMode=OFF`, and 0/40 with a 3-line ADIOS2 fix (store
  the timestep in the request handle and match preloads against it). The
  same fix runs the PCMS 4 -> 1 tests 45/45 clean under AddressSanitizer
  with preload on. Workaround here: `SpeculativePreloadMode=OFF` in the SST
  parameters of the tests and the demo; the SST N -> 1 ctests are back and
  passed 10 repeats each, and the 8 -> 1 demo 3/3 bit for bit (about 350
  ms/step instead of 300: data is pulled per step instead of pushed). Any
  PCMS user running several SST writers into a small reader with default
  parameters is exposed, C++ included (`AddApplication`'s default
  parameters do not set it).

  Reported upstream with the reproducer and fix as ADIOS2#5249
  (https://github.com/ornladios/ADIOS2/issues/5249, 2026-10-30). Decision:
  keep `SpeculativePreloadMode=OFF` in the demo and tests, no redev/PCMS
  default change; once a fixed ADIOS2 release is out, bump the minimum
  ADIOS2 version and drop the parameter.

  Separately, the container's 64 MB `/dev/shm` fills up with MPICH segments
  left by killed runs and then every run dies with a bus error at import;
  start the container with a larger `--shm-size`.
- The prototype had drifted from the bindings: `from_mesh` was passed
  `CoordinateSystem.Cartesian` (the factory, not a system) and
  `get_dof_holder_data()` now always has a component axis. Fixed.
- The EMPIRE step height is 5 mm, not the synthetic 3.3 mm; 600 x 100 frames
  of it need the per-pixel coverage of 3.5 (not implemented).

Not done: per-pixel coverage (3.5) for non-conforming grids; a live view;
S7; binding `Mesh::migrate` to distribute a mesh by a given partition;
bumping the minimum ADIOS2 version and dropping `SpeculativePreloadMode=OFF`
once ADIOS2#5249 is fixed in a release.

### 8.8 Comparison with the training study

| | Streaming demo | Training study (section 7) |
|---|---|---|
| Claim | PCMS ingests live, decomposed simulation data into ML, conservatively, over ADIOS2 | Conservative ingestion gives better or cheaper surrogates |
| PCMS features shown | Coupler (incl. redistribution), ADIOS2/SST, conservative transfer | Conservative transfer |
| Trained model needed | No (pretrained) | Yes, several |
| Exodus data needed | 1 run; none to start (S3) | 1-2 full runs before anything starts |
| GPU cost | A few node-hours | ~50-100 node-hours (pending benchmark) |
| Critical path | S1 bindings, then S7 SST on the cluster | Data transfer, then multi-day training |
| Type of risk | Engineering: early, visible, debuggable | Scientific: null/ambiguous result arrives late |
| Lasting value | Python coupler bindings for all PCMS Python users; in-situ ML template | Scientific result |

**Hands-on time for the project lead** (who reviews and runs on the
cluster), assuming code is written and tested off-cluster:

| | Training | Streaming (reduced scope) |
|---|---|---|
| Reviewing PRs | Converter | Converter + coupler bindings (larger) |
| Data onto the cluster | 1-2 full runs | 1 run; not needed at first |
| Cluster build | PyTorch env | PyTorch env + PCMS Python with ADIOS2 SST |
| Running | Launch 8 jobs, check on them over 3-4 days, rerun failures | Start the producer job and the consumer, record |
| Interpreting | Rollout comparison; null result means rethinking the talk | Works or visibly doesn't |
| Main unknown | Whether there is a measurable difference | SST behaviour on the cluster |

## 9. Validation

- A constant field maps to exactly 1 on in-domain nodes.
- Per step: the weighted tensor integral of n equals the mesh integral
  (~1e-14). Report the same quantity for the point-sampled pipeline.
- Minimum density >= 0 at every step for lumped/bounded. Count negative
  nodes for consistent.
- Compare electron count and centre of mass (`electron_diagnostics`)
  against EMPIRE's own diagnostics for both pipelines. Do **not** use
  training L1 as the success metric: in earlier PCMS ML-ingestion experiments,
  end-to-end learning error (~1e-2) swamped ingestion error.
- Reverse direction: PCMS can project decoder output back onto the EMPIRE
  mesh, for comparison in the simulation's own discretisation.

## 10. Open questions

1. ~~Exodus file location and layout~~ Answered for the test file (8.1):
   single joined file, 51 steps. Still open: where the full-cadence runs
   (all 12, with fields) live.
2. ~~Exodus variable names and centering~~ Answered for the test file
   (8.1): electron moments only, all cell-centred, no Ex/Ey/Bz. The
   channel order in section 2 stays unconfirmed until a file with fields
   is available.
3. SEACAS Omega_h rebuild vs the netCDF4 reader.
4. Wall alignment with grid lines (3.5).
5. A later frame with developed plasma, to see how density was originally
   sampled.
6. Keep velocity (checkpoint-compatible) or switch to momentum density
   (retrain).
7. Whether the Omega_h gmsh reader accepts a 2.2 file with no boundary
   elements and preserves entity order (`mesh_from_arrays` matches by
   location, so reordering is harmless).
8. Which batch size the 12-day run used (Table 1 says 8, notebook says 1),
   and on what hardware.
9. GPU availability (count and class) for the comparison runs.
10. Local mesh grading in the plasma region, which sets how coarse the grid
    can go (6.1).
11. ~~Whether PyOmega_h can give the producer a distributed mesh~~ Yes:
    `read_mesh_binary` on N ranks, then `balance()` (8.5).
12. ~~Whether SST `QueueFullPolicy=Discard` is safe with redev~~ No (8.9).
