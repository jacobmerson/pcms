"""Step transport between the producer (simulation) and consumer (ML).

LoopbackTransport is an in-process bounded queue used for development and
tests. It emulates the ADIOS2 SST queue: QueueLimit with QueueFullPolicy
"Block" (the producer waits) or "Discard" (the step being written is dropped).

PcmsTransport makes the same calls over the PCMS coupler (redev on ADIOS2
SST or BP4) between two programs: producer and consumer run as separate MPI
jobs.
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

    # Producer side.
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
        """End of stream. Never blocks: the consumer drains what is queued,
        then stops."""
        self._closed.set()

    # Consumer side.
    def messages(self, poll=0.05):
        while True:
            try:
                yield self._queue.get(timeout=poll)
            except queue.Empty:
                if self._closed.is_set() and self._queue.empty():
                    return


class PcmsTransport:
    """Steps over the PCMS coupler (redev on ADIOS2 SST or BP4).

    The consumer is the coupler server, on one rank; the producer is the
    client, on one or more ranks. Each side reads its mesh from disk: the
    consumer the whole mesh, a producer rank the whole mesh or its piece of a
    partition (partition.py). Entities are matched by the "global" tag, so
    pieces must carry the global IDs of the consumer's mesh and give each
    shared vertex a single owner (node_owned).

    Per step the producer sends every field in field_centering plus a small
    meta array [step, time]; step -1 marks the end of the stream.
    """

    def __init__(self, role, mesh, field_centering, path=".",
                 transport="SST", params=None, node_owned=None, comm=None):
        import pcms
        from geometry import lagrange_space

        if role not in ("producer", "consumer"):
            raise ValueError(f"unknown role {role!r}")
        self.role = role
        self.sent = 0
        self.dropped = 0
        server = role == "consumer"
        # One server rank owns the whole mesh: a cut tree with a single leaf.
        partition = pcms.RCBPartition(2, [0], [0.0]) if server else None
        transport_type = pcms.TransportType.__members__[transport]
        if params is None and transport == "SST":
            params = SST_PARAMS
        self._coupler = pcms.CouplerComm("mitl", comm, server, partition)
        self._app = self._coupler.add_application(
            "empire", path=path, transport=transport_type, params=params)
        orders = {"elem": 0, "node": 1}
        owned = {"elem": None, "node": node_owned}
        self._spaces = {
            c: lagrange_space(mesh, orders[c], layout_name=c,
                              owned_mask=owned[c])
            for c in sorted(set(field_centering.values()))}
        # Sorted so both sides register (and exchange) in the same order.
        self._handles = {
            name: self._app.add_field(name, self._spaces[c])
            for name, c in sorted(field_centering.items())}
        self._meta = np.zeros(2)
        self._meta_handle = self._app.add_data("meta", self._meta)

    # Producer side.
    def send(self, message):
        for name, handle in self._handles.items():
            handle.field.set_dof_holder_data(
                np.ascontiguousarray(message.fields[name], np.float64))
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

    # Consumer side.
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
                name: np.array(h.field.get_dof_holder_data()[:, 0])
                for name, h in self._handles.items()})


# ADIOS2 SST engine parameters for the PCMS transport. Start with "Block":
# redev exchanges setup messages over the same channel, and discarding those
# would break the handshake (to verify before switching to "Discard").
SST_PARAMS = {
    "OpenTimeoutSecs": "300",
    "DataTransport": "WAN",
    "QueueLimit": "4",
    "QueueFullPolicy": "Block",
}
