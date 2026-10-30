"""Two-process field exchange through the Python coupler bindings.

Run as two MPI jobs (see test/CMakeLists.txt):

    coupler_exchange.py client <BP4|SST>
    coupler_exchange.py server <BP4|SST>

The client sends a P0 and a P1 field whose values encode each DOF holder's
global ID, plus a small data array; the server receives them and checks
arrival bit for bit. Then the server sends the fields back so both
directions are covered. Exits nonzero on a mismatch.
"""

import gc
import sys

import numpy as np

import pcms
import PyOmega_h as omega_h

# ADIOS2 SST parameters; BP4 uses the coupler defaults.
# SpeculativePreloadMode OFF: works around an ADIOS2 SST bug (2.12.1 and
# master) that crashes or misreads N -> 1 with preload on, which is the default
# for one reader rank: https://github.com/ornladios/ADIOS2/issues/5249
# Drop this once the minimum ADIOS2 version has the fix.
SST_PARAMS = {"OpenTimeoutSecs": "60", "DataTransport": "WAN",
              "SpeculativePreloadMode": "OFF"}


def expected(field, sign):
    """Values tied to the DOF holder locations, so they survive any local
    ordering: sign * (1 + x + 10 y) per DOF holder."""
    coords = np.asarray(field.get_dof_holder_coordinates())
    return sign * (1.0 + coords[:, 0] + 10.0 * coords[:, 1])


def run(role, transport_name):
    is_server = role == "server"
    lib = omega_h.OmegaHLibrary()
    mesh = omega_h.build_box(lib.world(), omega_h.Family.SIMPLEX, 1.0, 1.0,
                             0.0, 8, 8, 0, False)
    transport = pcms.TransportType.__members__[transport_name]
    params = SST_PARAMS if transport == pcms.TransportType.SST else None

    partition = pcms.RCBPartition(2, [0], [0.0]) if is_server else None
    coupler = pcms.CouplerComm("coupler_exchange", None, is_server, partition)
    app = coupler.add_application("exchange", transport=transport,
                                  params=params)
    spaces = {
        order: pcms.LagrangeFunctionSpace.from_mesh(
            mesh, order, backend=pcms.LagrangeFunctionSpace.Backend.OmegaH,
            layout_name=f"p{order}")
        for order in (0, 1)
    }
    handles = {order: app.add_field(f"u{order}", space)
               for order, space in spaces.items()}
    meta = np.zeros(2)
    meta_handle = app.add_data("meta", meta)

    def send(sign):
        for h in handles.values():
            h.field.set_dof_holder_data(expected(h.field, sign))
        meta[:] = (7.0, sign * 0.25)
        app.begin_send_phase()
        for h in handles.values():
            h.send()
        meta_handle.send()
        app.end_send_phase()

    def receive_and_check(sign):
        meta[:] = 0.0
        app.begin_receive_phase()
        for h in handles.values():
            h.receive()
        meta_handle.receive()
        app.end_receive_phase()
        ok = True
        for order, h in handles.items():
            got = np.asarray(h.field.get_dof_holder_data())[:, 0]
            want = expected(h.field, sign)
            if not np.array_equal(got, want):
                print(f"{role}: P{order} mismatch, max err "
                      f"{np.abs(got - want).max()}")
                ok = False
        if not np.array_equal(meta, [7.0, sign * 0.25]):
            print(f"{role}: meta mismatch {meta}")
            ok = False
        return ok

    if is_server:
        ok = receive_and_check(1.0)
        send(-1.0)
    else:
        send(1.0)
        ok = receive_and_check(-1.0)
    print(f"{role} ({transport_name}): {'passed' if ok else 'FAILED'}")

    # Release pcms/Omega_h objects before the library.
    handles = meta_handle = app = coupler = spaces = mesh = None
    gc.collect()
    del lib
    gc.collect()
    return ok


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1], sys.argv[2]) else 1)
