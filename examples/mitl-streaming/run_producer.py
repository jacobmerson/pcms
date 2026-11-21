#!/usr/bin/env python3
"""Producer job: replay a run and stream it over the PCMS coupler.

    mpiexec -n N python run_producer.py --source exodus --exo flow.exo
"""

import argparse

import numpy as np

import runtime
from partition import (PieceSource, balanced_mesh, merge_parts, mesh_piece,
                       strip_parts)
from producer import run_producer
from sources import add_source_args, make_source
from transport import SST_PARAMS, PcmsTransport


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_source_args(p)
    p.add_argument("--rate", type=float, default=None,
                   help="steps per second (default: unthrottled)")
    p.add_argument("--partition", choices=["balance", "source"],
                   default="balance",
                   help="N > 1: Omega_h-balanced distributed mesh, or serial "
                        "pieces of the source's decomposition")
    add_transport_args(p)
    return p.parse_args()


def add_transport_args(p):
    p.add_argument("--transport", choices=["SST", "BP4"], default="SST")
    p.add_argument("--path", default=".",
                   help="directory for the ADIOS2 contact/BP files")
    p.add_argument("--queue-limit", type=int, default=4)
    p.add_argument("--policy", choices=["block", "discard"], default="block",
                   help="SST QueueFullPolicy")


def transport_params(args):
    if args.transport != "SST":
        return None
    return dict(SST_PARAMS, QueueLimit=str(args.queue_limit),
                QueueFullPolicy=args.policy.capitalize())


def run(args):
    world = runtime.library().world()
    rank, size = world.rank(), world.size()
    node_owned = None
    if size == 1:
        source = make_source(args, world)
        mesh = source.mesh
    else:
        local = world.split(rank, 0)
        full = make_source(args, local if args.source == "synthetic" else None)
        if args.partition == "balance":
            piece = balanced_mesh(full.points, full.tris, world, args.path)
        else:
            parts = (merge_parts(full.proc_id(), size)
                     if args.source == "exodus"
                     else strip_parts(full.points, full.tris, size))
            piece = mesh_piece(full.points, full.tris, parts, rank, local)
        source = PieceSource(full, piece)
        mesh, node_owned = piece.mesh, piece.node_owned
        owned = (node_owned if node_owned is not None
                 else np.asarray(mesh.owned(0)))
        print(f"producer rank {rank}/{size}: {mesh.nelems()} elements, "
              f"{int(owned.sum())} of {mesh.nverts()} vertices owned",
              flush=True)
    transport = PcmsTransport("producer", mesh, source.field_centering,
                              path=args.path, transport=args.transport,
                              params=transport_params(args),
                              node_owned=node_owned)
    run_producer(source, transport, args.rate)
    if rank == 0:
        print(f"producer: sent {transport.sent} steps", flush=True)


def main():
    run(parse_args())
    runtime.shutdown()


if __name__ == "__main__":
    main()
