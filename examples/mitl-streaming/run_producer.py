#!/usr/bin/env python3
"""Producer job: replay a run and stream it over the PCMS coupler.

    mpiexec -n N python run_producer.py --source exodus --exo flow.exo

With N > 1 each rank sends only its piece of the mesh: EMPIRE's own
decomposition (PROC_ID, merged to N parts) for Exodus input, x strips for
the synthetic source. Start run_consumer.py (one rank) alongside; see
run_pcms_demo.sh.
"""

import argparse

import runtime
from partition import (PieceSource, merge_parts, mesh_piece, strip_parts)
from producer import run_producer
from sources import add_source_args, make_source
from transport import SST_PARAMS, PcmsTransport


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_source_args(p)
    p.add_argument("--rate", type=float, default=None,
                   help="steps per second (default: unthrottled)")
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
        # Every rank reads the whole run (cheap at this size) and builds a
        # serial mesh of its own piece.
        local = world.split(rank, 0)
        if args.source == "exodus":
            full = make_source(args, None)
            parts = merge_parts(full.proc_id(), size)
        else:
            full = make_source(args, local)
            parts = strip_parts(full.points, full.tris, size)
        piece = mesh_piece(full.points, full.tris, parts, rank, local)
        source = PieceSource(full, piece)
        mesh, node_owned = piece.mesh, piece.node_owned
        print(f"producer rank {rank}/{size}: {mesh.nelems()} elements, "
              f"{int(node_owned.sum())} of {mesh.nverts()} vertices owned",
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
