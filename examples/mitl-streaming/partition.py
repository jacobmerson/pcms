"""Per-rank meshes for a partitioned producer (plan 8.5, task P1).

Two ways to give each producer rank its part of the full mesh:

balanced_mesh (the default): a distributed Omega_h mesh. Rank 0 writes the
full mesh in Omega_h's binary format, all ranks read it and balance() spreads
it over them. Omega_h keeps the file order as the global numbering and
decides vertex ownership itself.

mesh_piece: a serial mesh of a given partition (EMPIRE's PROC_ID, or x
strips), one per rank on a single-rank communicator. The pieces carry the
full-mesh numbering in the "global" tag and a vertex on a piece boundary is
owned by the lowest part that touches it.

Either way the "global" tag is the element and vertex index in the full mesh
(the Exodus file order), which is how the coupler matches the consumer's
mesh, and every DOF is sent exactly once.
"""

import os
import shutil
from dataclasses import dataclass

import numpy as np
import PyOmega_h as omega_h

from geometry import compact, mesh_from_arrays


def strip_parts(points, tris, nparts):
    """Element -> part by centroid x, in nparts equal-count strips (the
    partition for the synthetic source)."""
    order = np.argsort(points[tris].mean(axis=1)[:, 0], kind="stable")
    parts = np.empty(len(tris), dtype=np.int64)
    for part, chunk in enumerate(np.array_split(order, nparts)):
        parts[chunk] = part
    return parts


def merge_parts(parts, nparts):
    """Merge an existing decomposition (e.g. EMPIRE's PROC_ID over 8 ranks)
    into nparts consecutive groups."""
    nold = int(parts.max()) + 1
    if nparts > nold:
        raise ValueError(f"cannot split {nold} parts into {nparts}")
    return parts * nparts // nold


def vertex_owners(tris, parts, nverts):
    """Owning part of each vertex: the lowest part among its elements."""
    owner = np.full(nverts, np.iinfo(np.int64).max)
    np.minimum.at(owner, tris.ravel(), np.repeat(parts, tris.shape[1]))
    return owner


@dataclass
class MeshPiece:
    mesh: object
    elem_gids: np.ndarray     # local element -> full-mesh element
    vert_gids: np.ndarray     # local vertex -> full-mesh vertex
    node_owned: np.ndarray    # int8 per local vertex, 1 = this rank sends
                              # it; None = the mesh's own ownership


def balanced_mesh(points, tris, world, directory):
    """The full mesh (points, tris) distributed over world by Omega_h's
    balance() (recursive inertial bisection). directory must be visible to
    every rank; the mesh file is removed again."""
    rank = world.rank()
    path = os.path.join(directory, "mitl_source_mesh.osh")
    local = world.split(rank, 0)
    if rank == 0:
        omega_h.write_mesh_binary(path, mesh_from_arrays(points, tris, local))
    world.barrier()
    # A one-part file read on N ranks: the elements land on rank 0.
    mesh = omega_h.read_mesh_binary(path, world)
    mesh.balance()
    world.barrier()
    if rank == 0:
        shutil.rmtree(path)
    return MeshPiece(mesh, np.asarray(mesh.globals(2)),
                     np.asarray(mesh.globals(0)), None)


def mesh_piece(points, tris, parts, part, comm):
    """The piece of the full mesh (points, tris) with parts == part, as a
    serial mesh on comm (a single-rank communicator)."""
    keep = parts == part
    piece_points, piece_tris, vert_gids = compact(points, tris, keep)
    elem_gids = np.flatnonzero(keep)
    mesh = mesh_from_arrays(piece_points, piece_tris, comm)
    mesh.set_tag(2, "global", elem_gids.astype(np.int64))
    mesh.set_tag(0, "global", vert_gids.astype(np.int64))
    owners = vertex_owners(tris, parts, len(points))
    node_owned = (owners[vert_gids] == part).astype(np.int8)
    return MeshPiece(mesh, elem_gids, vert_gids, node_owned)


class PieceSource:
    """A source restricted to one mesh piece: same interface, fields hold
    only the piece's elements and vertices."""

    def __init__(self, source, piece):
        self._source = source
        self._index = {"elem": piece.elem_gids, "node": piece.vert_gids}
        self.mesh = piece.mesh
        self.num_steps = source.num_steps
        self.field_centering = source.field_centering

    def time(self, step):
        return self._source.time(step)

    def fields(self, step):
        return {name: values[self._index[self.field_centering[name]]]
                for name, values in self._source.fields(step).items()}
