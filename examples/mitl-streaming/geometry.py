"""Hardcoded MITL-like domain, pixel grid and mesh construction helpers.

The domain is an L-shape standing in for the 2D inner MITL of the hnode
plasma-gun example (arXiv:2604.21101, Fig. 6): a 50 x 8.25 mm box with the
lower-left corner below the backward-facing step removed. The step is placed
on grid lines of the 600 x 100 frame grid so that the domain-matched target
mesh conforms exactly (no partially covered pixels).
"""

import os
import tempfile
from dataclasses import dataclass

import numpy as np

import pcms
import PyOmega_h as omega_h

LENGTH = 50.0
HEIGHT = 8.25
STEP_X = 35.0
STEP_Y = 3.3
# The EMPIRE MITL mesh (pcms_testcases exodus/flow/flow.exo) is the same L
# shape with the step at y = 5 mm.
EMPIRE_STEP_Y = 5.0


def inside(points, step_y=STEP_Y):
    """True for points in the closed L-shaped domain."""
    x, y = points[:, 0], points[:, 1]
    in_box = (x >= 0.0) & (x <= LENGTH) & (y >= 0.0) & (y <= HEIGHT)
    return in_box & ~((x < STEP_X) & (y < step_y))


@dataclass(frozen=True)
class Grid:
    """nx x ny pixels over the domain bounding box; frames are node-valued,
    shape (ny + 1, nx + 1), indexed [iy, ix] with iy increasing with y."""

    nx: int = 600
    ny: int = 100
    x0: float = 0.0
    y0: float = 0.0
    x1: float = LENGTH
    y1: float = HEIGHT
    step_y: float = STEP_Y

    @property
    def dx(self):
        return (self.x1 - self.x0) / self.nx

    @property
    def dy(self):
        return (self.y1 - self.y0) / self.ny

    @property
    def shape(self):
        return (self.ny + 1, self.nx + 1)

    def node_coords(self):
        X, Y = np.meshgrid(self.x0 + np.arange(self.nx + 1) * self.dx,
                           self.y0 + np.arange(self.ny + 1) * self.dy)
        return np.stack([X.ravel(), Y.ravel()], 1)

    def check_step_aligned(self):
        """The domain-matched target mesh is only exact if the step lies on
        grid lines."""
        for value, origin, h in ((STEP_X, self.x0, self.dx),
                                 (self.step_y, self.y0, self.dy)):
            k = (value - origin) / h
            if abs(k - round(k)) > 1e-9:
                raise ValueError(f"step at {value} is not on a grid line "
                                 f"(spacing {h})")

    def inside(self, points):
        return inside(points, self.step_y)


def structured_triangles(x, y):
    """Two triangles per cell of the tensor grid x by y. Returns (points,
    tris) with points ordered x-fastest."""
    nx, ny = len(x) - 1, len(y) - 1
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], 1)
    iy, ix = np.mgrid[0:ny, 0:nx]
    n00 = (iy * (nx + 1) + ix).ravel()
    n10, n01 = n00 + 1, n00 + nx + 1
    n11 = n01 + 1
    tris = np.concatenate([np.stack([n00, n10, n11], 1),
                           np.stack([n00, n11, n01], 1)])
    return points, tris


def compact(points, tris, keep):
    """Drop triangles not in `keep` and the vertices they no longer use.
    Returns (points, tris, used) where used maps new -> old vertex ids."""
    tris = tris[keep]
    used = np.unique(tris)
    local = np.full(len(points), -1)
    local[used] = np.arange(len(used))
    return points[used], local[tris], used


def mesh_from_arrays(points, tris, comm):
    """Omega_h 2D triangle mesh via a gmsh 2.2 round trip (PyOmega_h has no
    build-from-arrays binding). The reader preserves vertex and element
    order; this is checked so callers can rely on it."""
    fd, path = tempfile.mkstemp(suffix=".msh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
            f.write(f"$Nodes\n{len(points)}\n")
            for i, (px, py) in enumerate(points, 1):
                f.write(f"{i} {px:.17g} {py:.17g} 0\n")
            f.write(f"$EndNodes\n$Elements\n{len(tris)}\n")
            for i, (a, b, c) in enumerate(tris, 1):
                f.write(f"{i} 2 2 1 1 {a + 1} {b + 1} {c + 1}\n")
            f.write("$EndElements\n")
        mesh = omega_h.read_mesh_gmsh(path, comm)
    finally:
        os.unlink(path)
    if not (np.allclose(mesh.coords().reshape(-1, 2), points)
            and np.array_equal(mesh.ask_elem_verts().reshape(-1, 3), tris)):
        raise RuntimeError("gmsh round trip reordered mesh entities")
    return mesh


def source_mesh(comm, nx_left=170, nx_right=180, ny_low=40, ny_high=60,
                jitter=0.25, seed=0):
    """Graded, jittered triangle mesh of the L-shaped domain.

    Stands in for the EMPIRE mesh (~55k triangles by default, like the
    paper's): mesh lines pass through the step, the right section is finer
    than the left, and interior vertices are jittered so mesh cells do not
    align with frame pixels.
    """
    x = np.unique(np.concatenate([np.linspace(0.0, STEP_X, nx_left + 1),
                                  np.linspace(STEP_X, LENGTH, nx_right + 1)]))
    y = np.unique(np.concatenate([np.linspace(0.0, STEP_Y, ny_low + 1),
                                  np.linspace(STEP_Y, HEIGHT, ny_high + 1)]))
    points, tris = structured_triangles(x, y)
    points, tris, _ = compact(points, tris, inside(points[tris].mean(axis=1)))

    rng = np.random.default_rng(seed)
    fixed = _boundary_or_step(points)
    hx, hy = np.diff(x).min(), np.diff(y).min()
    shift = rng.uniform(-jitter, jitter, points.shape) * (hx, hy)
    points = np.where(fixed[:, None], points, points + shift)
    return mesh_from_arrays(points, tris, comm)


def _boundary_or_step(points):
    x, y = points[:, 0], points[:, 1]
    tol = 1e-12
    return ((np.abs(x) < tol) | (np.abs(x - LENGTH) < tol) | (np.abs(y) < tol)
            | (np.abs(y - HEIGHT) < tol) | (np.abs(x - STEP_X) < tol)
            | (np.abs(y - STEP_Y) < tol))


def lagrange_space(mesh, order, layout_name="", owned_mask=None):
    """P0 (order 0) or P1 scalar space; Cartesian coordinates. layout_name
    is needed to exchange the space's fields through the coupler."""
    return pcms.LagrangeFunctionSpace.from_mesh(
        mesh, order, backend=pcms.LagrangeFunctionSpace.Backend.OmegaH,
        layout_name=layout_name, owned_mask=owned_mask)
