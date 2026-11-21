"""P0, P1 and data exchange both ways between two programs over the coupler.

Usage: coupler_exchange.py client|server BP4|SST (two MPI jobs).
"""

import gc
import sys

import numpy as np

import pcms
import PyOmega_h as omega_h
from pcms import coupler

# ADIOS2 SST parameters; BP4 uses the coupler defaults.
SST_PARAMS = {"OpenTimeoutSecs": "60", "DataTransport": "WAN"}


def expected(field, sign):
    """Return sign * (1 + x + 10 y) at the field's DOF holders."""
    coords = field.get_layout().get_dof_holder_coordinates().get_values()
    return sign * (1.0 + coords[:, 0] + 10.0 * coords[:, 1])


def run(role, transport_name):
    is_server = role == "server"
    lib = omega_h.OmegaHLibrary()
    mesh = omega_h.build_box(lib.world(), omega_h.Family.SIMPLEX, 1.0, 1.0,
                             0.0, 8, 8, 0, False)
    transport = coupler.TransportType.__members__[transport_name]
    params = SST_PARAMS if transport == coupler.TransportType.SST else None

    partition = coupler.RCBPartition(2, [0], [0.0]) if is_server else None
    comm = coupler.CouplerComm("coupler_exchange", None, is_server, partition)
    app = comm.add_application("exchange", transport=transport,
                               params=params)
    geometry = pcms.omega_h.create_geometry(mesh)
    spaces = {
        order: pcms.create_lagrange_space(
            geometry, order, 1, backend=pcms.LagrangeBackend.OmegaH,
            layout_name=f"p{order}")
        for order in (0, 1)
    }
    handles = {order: app.add_function(f"u{order}", space)
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
            got = h.field.get_dof_holder_data().get_values()[:, 0]
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
    handles = meta_handle = app = comm = spaces = geometry = mesh = None
    gc.collect()
    del lib
    gc.collect()
    return ok


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1], sys.argv[2]) else 1)
