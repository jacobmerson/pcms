#!/usr/bin/env python3
"""moving-points: tracing field lines with a batch of tracers.

Many codes evaluate a field at points that change at every step: particles
pushed in a field, field lines followed through a magnetic field, probes that
sweep a mesh. Locating a point on a mesh is the expensive part of evaluating
a field there, so PCMS separates it: QueryPoints locates the points, keeps the
result, and reuses its storage when you locate the next set of points. This is
the Python twin of main.cpp.

A batch of tracers follows the field lines of a rotation field

    v(x, y) = (-(y - 1/2), x - 1/2)

on the unit square, stored as a P1 vector field on a triangle mesh. The field
lines are circles around the center, traversed once in 2 pi. Each tracer is
advanced by the classical Runge-Kutta method (RK4), which evaluates the field
at four sets of points per step: the tracers advanced by the previous stage.

Tracers on small circles stay in the domain and come back to where they
started. Tracers on a large circle leave the domain, where the field is not
defined: with the FILL policy their points are reported as missing, and we
stop advancing them.

  1. the mesh, the vector field and the tracers;
  2. one RK4 stage: locate, then evaluate, then read missing_points;
  3. the integration loop over the stages;
  4. the checks: closed circles inside, tracers lost outside.
"""
import pathlib
import sys

import numpy as np
import pcms
import PyOmega_h as omega_h

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from example_utils import Report  # noqa: E402

CENTER = 0.5
# RK4 with 200 steps per revolution has a global error of about
# (2 pi / 200)^4 / 120 of the radius per revolution, far below this.
CLOSURE_TOL = 1e-6
STEPS_PER_REVOLUTION = 200


class Tracers:
    """A batch of tracers advanced through a field. Each call of velocity is
    one stage of RK4."""

    def __init__(self, geometry, velocity, start):
        self.velocity_field = velocity
        self.position = start.copy()
        self.lost = np.zeros(len(start), dtype=bool)
        self.num_missing_last = 0
        # Points outside the domain get the velocity 0 instead of an error.
        self.query = pcms.QueryPoints(
            geometry, pcms.OutOfBoundsPolicy(pcms.OutOfBoundsMode.FILL, 0.0))

    def velocity(self, points):
        """The velocity at `points`: locate finds the element of every point
        and evaluate interpolates the vertex values in it. QueryPoints.locate
        keeps the storage of the previous call when the number of points does
        not grow. Points outside the mesh are not evaluated: their rows of the
        result hold the fill value, and missing_points lists them."""
        self.query.locate(pcms.CoordinateView(points))
        out = np.zeros((len(points), 2))
        self.query.evaluate(
            self.velocity_field,
            pcms.ValueView(self.velocity_field.get_data().get_value_basis(),
                           out))
        missing = self.query.bound.missing_points()
        self.num_missing_last = len(missing)
        self.lost[missing] = True
        return out

    def step(self, dt):
        """One RK4 step of size dt for the tracers that are not lost."""
        k1 = self.velocity(self.position)
        k2 = self.velocity(self.position + 0.5 * dt * k1)
        k3 = self.velocity(self.position + 0.5 * dt * k2)
        k4 = self.velocity(self.position + dt * k3)
        advance = dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # a tracer that left the domain stays where it was
        self.position[~self.lost] += advance[~self.lost]


def main():
    report = Report()

    # ---------------------------------------------------------------------
    # 1. Mesh, vector field and tracers
    #
    # The field has two components, so it is a vector: the values are tagged
    # as a vector in the Cartesian system of the mesh, which tells PCMS how
    # the components change when the field is re-expressed in other
    # coordinates (see coordinate-systems). Its values at the vertices are
    # set from v; a linear field is exact on a P1 mesh.
    lib = omega_h.OmegaHLibrary()
    mesh = omega_h.build_box(lib.world(), omega_h.Family.SIMPLEX, 1.0, 1.0,
                             0.0, 16, 16, 0, False)
    geometry = pcms.backend.omega_h.create_geometry(
        mesh, pcms.csys.Cartesian.create(2))
    space = pcms.create_lagrange_space(
        geometry, 1, 2, backend=pcms.LagrangeBackend.OmegaH)
    velocity = space.create_function("velocity", pcms.values.Vector)
    holders = velocity.get_layout().get_dof_holder_coordinates().get_values()
    velocity.set_dof_holder_data(pcms.ValueView(
        velocity.get_data().get_value_basis(),
        np.stack([-(holders[:, 1] - CENTER), holders[:, 0] - CENTER],
                 axis=1)))

    # Tracers start on circles around the center: three small ones, which
    # stay in the unit square, and a large one, which does not.
    rings = [(0.15, 8), (0.25, 8), (0.35, 8), (0.60, 12)]
    radius, ring_of, start = [], [], []
    for index, (r, count) in enumerate(rings):
        for k in range(count):
            angle = 0.1 + 2.0 * np.pi * k / count
            start.append([CENTER + r * np.cos(angle),
                          CENTER + r * np.sin(angle)])
            radius.append(r)
            ring_of.append(index)
    start = np.array(start)
    radius = np.array(radius)
    ring_of = np.array(ring_of)
    tracers = Tracers(geometry, velocity, start)

    # ---------------------------------------------------------------------
    # 2. One stage: found and missing points
    #
    # The first stage evaluates the field at the starting points. The large
    # circle has 12 tracers; those that start outside the unit square are
    # missing. We compare with the answer from the geometry of the circle.
    tracers.velocity(tracers.position)
    outside = int(np.sum(np.max(np.abs(start - CENTER), axis=1) > 0.5))
    print("%d tracers, %d of them start outside the unit square "
          "(%d missing points)"
          % (len(start), outside, tracers.num_missing_last))
    report.check("first stage: MissingPoints are the tracers outside the mesh",
                 tracers.num_missing_last == outside,
                 tracers.num_missing_last)
    # Tracers that start outside have nothing to trace: they are lost from
    # the beginning. The others must not be marked by this stage.
    report.check("first stage: lost tracers are those that start outside",
                 int(np.sum(tracers.lost)) == outside,
                 int(np.sum(tracers.lost)))

    # ---------------------------------------------------------------------
    # 3. The integration loop
    #
    # One revolution takes 2 pi (the angular velocity is 1). A tracer that is
    # lost in a stage, because the stage's point is outside the mesh, stops.
    dt = 2.0 * np.pi / STEPS_PER_REVOLUTION
    for _ in range(STEPS_PER_REVOLUTION):
        tracers.step(dt)

    # ---------------------------------------------------------------------
    # 4. The checks
    #
    # The tracers inside come back to their start, on their circle. Those on
    # the large circle all leave the domain, so all are lost.
    inside = ring_of < len(rings) - 1
    closure = np.max(np.linalg.norm(
        tracers.position[inside] - start[inside], axis=1))
    radial = np.max(np.abs(np.linalg.norm(
        tracers.position[inside] - CENTER, axis=1) - radius[inside]))
    print("after one revolution: closure error %.3e, radius error %.3e"
          % (closure, radial))
    report.check("circles inside the square: no tracer is lost",
                 not np.any(tracers.lost[inside]),
                 int(np.sum(tracers.lost[inside])))
    report.check_at_most("circles inside the square: back at the start",
                         closure, CLOSURE_TOL)
    report.check_at_most("circles inside the square: radius conserved",
                         radial, CLOSURE_TOL)
    report.check("circle outside the square: every tracer is lost",
                 np.all(tracers.lost[~inside]),
                 int(np.sum(tracers.lost[~inside])))

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
