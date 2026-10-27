"""Consumer: conservative ingestion of streamed steps + surrogate in the loop.

For each received step the consumer rasterizes the mesh fields to a frame,
records the exact electron count, and compares against the current forecast.
A forecast is made from the latest three consecutive frames and re-anchored
every `anchor_every` steps. A gap in step numbers (a discarded step) resets
the window, since the model assumes consecutive frames.
"""

import collections
from dataclasses import dataclass

import numpy as np

from rasterizer import frame_channels, make_frame


@dataclass
class StepRecord:
    step: int
    time: float
    count: float            # sum(weights * n_e) of the ingested frame
    source_count: float     # exact sum(area * n_e) on the source mesh
    forecast_lead: int      # steps since the forecast's newest input; 0 = none
    density_error: float    # relative L2 of forecast density; nan if none
    forecast_count: float   # electron count of the forecast frame; nan if none


class Consumer:

    def __init__(self, rasterizer, model, anchor_every=20, horizon=None,
                 keep_frames=(), centering=None):
        self.rast = rasterizer
        self.centering = centering
        self.density_channel = None
        self.model = model
        self.anchor_every = anchor_every
        self.horizon = horizon or anchor_every
        if self.horizon < self.anchor_every:
            raise ValueError("horizon must cover the re-anchor interval")
        self.keep_frames = set(keep_frames)
        self.records = []
        self.frames = {}          # step -> (truth, forecast or None)
        self.anchors = []
        self.gaps = 0
        self._window = collections.deque(maxlen=3)
        self._forecast = None
        self._anchor_step = None

    def process(self, message):
        frame = make_frame(self.rast, message.fields, self.centering)
        if self.density_channel is None:
            self.density_channel = frame_channels(message.fields).index("n_e")
        if self._window and message.step != self._window[-1][0] + 1:
            self.gaps += 1
            self._window.clear()
            self._forecast = None
        self._window.append((message.step, frame))

        pred, lead = self._prediction(message.step)
        dens = frame[self.density_channel]
        if pred is not None:
            p = pred[self.density_channel]
            norm = np.linalg.norm(dens[self.rast.mask])
            err = (float(np.linalg.norm((p - dens)[self.rast.mask]) / norm)
                   if norm > 0 else float("nan"))
            fcount = self.rast.integral(p)
        else:
            err, fcount = float("nan"), float("nan")

        record = StepRecord(
            step=message.step, time=message.time,
            count=self.rast.integral(dens),
            source_count=self.rast.source_integral(message.fields["n_e"]),
            forecast_lead=lead, density_error=err, forecast_count=fcount)
        self.records.append(record)
        if message.step in self.keep_frames:
            self.frames[message.step] = (frame, pred)

        self._maybe_anchor(message.step)
        return record

    def _prediction(self, step):
        if self._forecast is None:
            return None, 0
        lead = step - self._anchor_step
        if not 1 <= lead <= len(self._forecast):
            return None, 0
        pred = np.where(self.rast.mask, self._forecast[lead - 1], 0.0)
        return pred, lead

    def _maybe_anchor(self, step):
        if len(self._window) < 3:
            return
        if (self._forecast is not None
                and step - self._anchor_step < self.anchor_every):
            return
        frames = np.stack([f for _, f in self._window])
        self._forecast = self.model.rollout(frames, self.horizon)
        self._anchor_step = step
        self.anchors.append(step)

    def run(self, transport):
        for message in transport.messages():
            self.process(message)
        return self.records
