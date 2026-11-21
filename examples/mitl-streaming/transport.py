"""Step transport between producer and consumer: an in-process queue
(LoopbackTransport) or the PCMS coupler (PcmsTransport).
"""

import queue
import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepMessage:
    step: int
    time: float
    fields: dict = field(default_factory=dict)


class LoopbackTransport:

    def __init__(self, queue_limit=4, policy="block"):
        if policy not in ("block", "discard"):
            raise ValueError(f"unknown policy {policy!r}")
        self._queue = queue.Queue(maxsize=queue_limit)
        self._policy = policy
        self._closed = threading.Event()
        self.sent = 0
        self.dropped = 0

    def send(self, message):
        if self._policy == "block":
            self._queue.put(message)
        else:
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                self.dropped += 1
                return
        self.sent += 1

    def close(self):
        """End the stream; never blocks."""
        self._closed.set()

    def messages(self, poll=0.05):
        while True:
            try:
                yield self._queue.get(timeout=poll)
            except queue.Empty:
                if self._closed.is_set() and self._queue.empty():
                    return


class PcmsTransport:
    """Steps over the PCMS coupler; the consumer is the server on one rank.

    Args:
        role: "producer" or "consumer".
        mesh: this rank's Omega_h mesh, with the full mesh's "global" tag.
        field_centering: field name -> "elem" / "node".
        node_owned: int8 per vertex, 1 = owned (serial pieces); None: the
            mesh's ownership.
    """

    def __init__(self, role, mesh, field_centering, path=".",
                 transport="SST", params=None, node_owned=None, comm=None):
        import pcms
        from geometry import lagrange_space, mesh_geometry

        if role not in ("producer", "consumer"):
            raise ValueError(f"unknown role {role!r}")
        self.role = role
        self.sent = 0
        self.dropped = 0
        server = role == "consumer"
        partition = (pcms.coupler.RCBPartition(2, [0], [0.0]) if server
                     else None)
        transport_type = pcms.coupler.TransportType.__members__[transport]
        if params is None and transport == "SST":
            params = SST_PARAMS
        self._coupler = pcms.coupler.CouplerComm("mitl", comm, server,
                                                 partition)
        self._app = self._coupler.add_application(
            "empire", path=path, transport=transport_type, params=params)
        orders = {"elem": 0, "node": 1}
        self._geometry = mesh_geometry(mesh, owned_mask=node_owned)
        self._spaces = {
            c: lagrange_space(self._geometry, orders[c], layout_name=c)
            for c in sorted(set(field_centering.values()))}
        self._owned = {c: owned_entities(mesh, c, node_owned, space)
                       for c, space in self._spaces.items()}
        self._centering = dict(field_centering)
        # sorted: both sides must register in the same order
        self._handles = {
            name: self._app.add_function(name, self._spaces[c])
            for name, c in sorted(field_centering.items())}
        self._meta = np.zeros(2)
        self._meta_handle = self._app.add_data("meta", self._meta, comm)

    def send(self, message):
        for name, handle in self._handles.items():
            owned = self._owned[self._centering[name]]
            handle.field.set_dof_holder_data(
                np.ascontiguousarray(message.fields[name][owned], np.float64))
        self._send_phase(message.step, message.time)
        self.sent += 1

    def close(self):
        """End of stream: the field contents are ignored by the consumer."""
        self._send_phase(-1, 0.0)

    def _send_phase(self, step, time):
        self._meta[:] = (step, time)
        self._app.begin_send_phase()
        for handle in self._handles.values():
            handle.send()
        self._meta_handle.send()
        self._app.end_send_phase()

    def messages(self):
        while True:
            self._app.begin_receive_phase()
            for handle in self._handles.values():
                handle.receive()
            self._meta_handle.receive()
            self._app.end_receive_phase()
            step, time = int(self._meta[0]), float(self._meta[1])
            if step < 0:
                return
            yield StepMessage(step, time, {
                name: np.array(
                    h.field.get_dof_holder_data().get_values()[:, 0])
                for name, h in self._handles.items()})


def owned_entities(mesh, centering, node_owned, space):
    """Return the mask of the local entities of `centering` this rank's fields
    on `space` hold; raise RuntimeError if they are not its DOF holders.
    """
    coords = np.asarray(mesh.coords()).reshape(-1, 2)
    if centering == "elem":
        owned = np.asarray(mesh.owned(2), dtype=bool)
        tris = np.asarray(mesh.ask_elem_verts()).reshape(-1, 3)
        coords = coords[tris].mean(axis=1)
    else:
        owned = np.asarray(mesh.owned(0) if node_owned is None else node_owned,
                           dtype=bool)
    holders = space.get_layout().get_dof_holder_coordinates().get_values()
    if not np.allclose(holders, coords[owned], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{centering} DOF holders are not the owned "
                           "entities in local order")
    return owned


# Block: redev's setup steps share the channel and must not be discarded
SST_PARAMS = {
    "OpenTimeoutSecs": "300",
    "DataTransport": "WAN",
    "QueueLimit": "4",
    "QueueFullPolicy": "Block",
}
