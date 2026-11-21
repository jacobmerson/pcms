"""N -> 1 exchange from a partitioned client to one server over the coupler.

Usage: coupler_partitioned.py client|server BP4|SST pieces|balanced. pieces:
serial per-rank meshes with global IDs and a vertex owned_mask; balanced: a
distributed Omega_h mesh. Values encode global IDs; the server checks them
bit for bit.
"""

import gc
import os
import shutil
import sys
import tempfile

import numpy as np

import pcms
import PyOmega_h as omega_h
from pcms import coupler

NX, NY = 24, 6
SST_PARAMS = {"OpenTimeoutSecs": "60", "DataTransport": "WAN"}


def structured_mesh():
    x, y = np.linspace(0.0, 2.0, NX + 1), np.linspace(0.0, 1.0, NY + 1)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.ravel(), Y.ravel()], 1)
    iy, ix = np.mgrid[0:NY, 0:NX]
    n00 = (iy * (NX + 1) + ix).ravel()
    n10, n01 = n00 + 1, n00 + NX + 1
    tris = np.concatenate([np.stack([n00, n10, n01 + 1], 1),
                           np.stack([n00, n01 + 1, n01], 1)])
    return points, tris


def mesh_from_arrays(points, tris, comm):
    """Return an Omega_h mesh of (points, tris) on comm, in the same order."""
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
    assert np.array_equal(mesh.ask_elem_verts().reshape(-1, 3), tris)
    return mesh


def values(gids, order):
    return 1000.0 * order + gids.astype(np.float64)


def run(role, transport_name, mode):
    is_server = role == "server"
    lib = omega_h.OmegaHLibrary()
    world = lib.world()
    rank, size = world.rank(), world.size()
    points, tris = structured_mesh()
    local = world.split(rank, 0)
    owned = None
    if is_server:
        mesh = mesh_from_arrays(points, tris, local)
        gids = {0: np.arange(len(tris)), 1: np.arange(len(points))}
    elif mode == "balanced":
        # In the working directory, which every rank sees; one name per test.
        path = f"coupler_partitioned_{transport_name}_{size}.osh"
        if rank == 0:
            omega_h.write_mesh_binary(path,
                                      mesh_from_arrays(points, tris, local))
        world.barrier()
        mesh = omega_h.read_mesh_binary(path, world)
        mesh.balance()
        world.barrier()
        if rank == 0:
            shutil.rmtree(path)
        gids = {0: np.asarray(mesh.globals(2)),
                1: np.asarray(mesh.globals(0))}
    else:
        # x strips of whole columns of cells, one per rank
        column = np.concatenate([np.tile(np.arange(NX), NY)] * 2)
        parts = column * size // NX
        keep = parts == rank
        used = np.unique(tris[keep])
        renumber = np.full(len(points), -1)
        renumber[used] = np.arange(len(used))
        mesh = mesh_from_arrays(points[used], renumber[tris[keep]], local)
        gids = {0: np.flatnonzero(keep), 1: used}
        mesh.set_tag(2, "global", gids[0].astype(np.int64))
        mesh.set_tag(0, "global", gids[1].astype(np.int64))
        owner = np.full(len(points), size)
        np.minimum.at(owner, tris.ravel(), np.repeat(parts, 3))
        owned = (owner[used] == rank).astype(np.int8)

    # fields hold the owned DOF holders only
    owned_dofs = {0: np.asarray(mesh.owned(2), dtype=bool),
                  1: np.asarray(mesh.owned(0) if owned is None else owned,
                                dtype=bool)}
    gids = {order: g[owned_dofs[order]] for order, g in gids.items()}

    transport = coupler.TransportType.__members__[transport_name]
    params = SST_PARAMS if transport == coupler.TransportType.SST else None
    partition = coupler.RCBPartition(2, [0], [0.0]) if is_server else None
    comm = coupler.CouplerComm(f"coupler_partitioned_{mode}", None, is_server,
                               partition)
    app = comm.add_application("partitioned", transport=transport,
                               params=params)
    geometry = pcms.omega_h.create_geometry(mesh, owned_mask=owned)
    handles = {}
    for order in (0, 1):
        space = pcms.create_lagrange_space(
            geometry, order, 1, backend=pcms.LagrangeBackend.OmegaH,
            layout_name=f"p{order}")
        handles[order] = app.add_function(f"u{order}", space)

    ok = True
    if is_server:
        app.begin_receive_phase()
        for h in handles.values():
            h.receive()
        app.end_receive_phase()
        for order, h in handles.items():
            got = h.field.get_dof_holder_data().get_values()[:, 0]
            if not np.array_equal(got, values(gids[order], order)):
                bad = np.flatnonzero(got != values(gids[order], order))
                print(f"server: P{order} wrong at {len(bad)} entities, "
                      f"e.g. {bad[:5]}")
                ok = False
    else:
        for order, h in handles.items():
            h.field.set_dof_holder_data(values(gids[order], order))
        app.begin_send_phase()
        for h in handles.values():
            h.send()
        app.end_send_phase()
    print(f"{role} rank {rank}/{size} ({transport_name}): "
          f"{'passed' if ok else 'FAILED'}")

    del handles, app, comm, geometry, mesh
    gc.collect()
    del local, world, lib
    gc.collect()
    return ok


if __name__ == "__main__":
    sys.exit(0 if run(*sys.argv[1:4]) else 1)
