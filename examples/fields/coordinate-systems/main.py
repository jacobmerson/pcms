#!/usr/bin/env python3
"""coordinate-systems: fields in cylindrical coordinates, queried in Cartesian.

Simulation codes use different coordinates: a tokamak code works in
(R, phi, Z), a particle code in (x, y, z). PCMS tags every set of coordinates
with a CoordinateSystem, and every vector or tensor value with the basis its
components are in, so that a field can be evaluated at points given in other
coordinates and come out in the components the caller asks for. This is the
Python twin of main.cpp.

Two ideas, both shown here:
  * the points: a CoordinateMap carries query points from the caller's
    coordinate system to the geometry's, and the evaluation happens at the
    mapped points;
  * the values: a scalar is the same number in every system, but the
    components of a vector are not; the map re-expresses them in the basis of
    the query points (the radial component of a vector is not its x
    component).

The geometry is a box mesh whose coordinates are (r, theta, z): r in [0, 2],
theta in [0, 1.6] rad, z in [0, 1.1]. A P1 field on it is linear in
(r, theta, z). We query it at Cartesian points with CartesianToCylindrical.

  1. the map on its own: mapped points, and values rotated to Cartesian;
  2. a scalar field evaluated at Cartesian points (QueryPoints with a map);
  3. a vector field whose components are rotated into the query's basis;
  4. the same through an Interpolator onto a Cartesian mesh, which binds the
     target's points once with Geometry.bind and the map;
  5. a field with cylindrical points but Cartesian components, for which the
     map passes the values through and only maps the points.
"""
import pathlib
import sys

import numpy as np
import pcms
import PyOmega_h as omega_h

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from example_utils import Report  # noqa: E402

ROUNDING_TOL = 1e-12
# With Cartesian components stored on a (r, theta, z) mesh the field is not
# linear in the mesh coordinates, so a P1 field only approximates it (section
# 5; the mesh has 8 x 8 x 6 cells).
BORROWED_BASIS_TOL = 2e-2


def vector_in(system):
    """The basis of a vector whose components are physical, the same unit in
    every direction (not scaled by the coordinates), in a given system."""
    return pcms.ValueBasis(system, pcms.ComponentScaling.Physical,
                           pcms.values.Vector)


def fill_vector(field, f):
    """Set the DOF holders of a 3-component field from f(p), where p are the
    coordinates of the holders in the field's system."""
    holders = field.get_layout().get_dof_holder_coordinates().get_values()
    field.set_dof_holder_data(
        pcms.ValueView(field.get_data().get_value_basis(), f(holders)))


def main():
    report = Report()
    lib = omega_h.OmegaHLibrary()
    cartesian = pcms.csys.Cartesian.create(3)
    cylindrical = pcms.csys.CylindricalRThetaZ.create()

    # ---------------------------------------------------------------------
    # 1. A coordinate map on its own
    #
    # CartesianToCylindrical maps points from Cartesian (x, y, z) to
    # cylindrical (r, theta, z). Binding it to a set of points does all the
    # per-point work once: the mapped points, and what is needed to rotate
    # components at them. The values run the other way: it turns a vector
    # given in cylindrical components into Cartesian ones, in place.
    #
    # Take e_theta, the unit vector of increasing theta, at three points. Its
    # cylindrical components are (0, 1, 0) everywhere; in Cartesian
    # components it is (-sin theta, cos theta, 0).
    xyz = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, -1.0], [1.0, 1.0, 0.0]])
    bound = pcms.CartesianToCylindrical().bind(
        pcms.CoordinateView(cartesian, xyz))
    rtz = bound.mapped_points().get_values()
    e_theta = np.tile([0.0, 1.0, 0.0], (3, 1))
    rotated = bound.transform_values(
        pcms.ValueView(vector_in(cylindrical), e_theta))
    expected = np.stack([-np.sin(rtz[:, 1]), np.cos(rtz[:, 1]),
                         np.zeros(3)], axis=1)
    error = max(np.max(np.abs(e_theta - expected)),
                np.max(np.abs(rtz[:, 0] - np.hypot(xyz[:, 0], xyz[:, 1]))))
    for p, e in zip(rtz, e_theta):
        print("  (r, theta, z) = (%.3f, %.3f, %.3f), e_theta in Cartesian "
              "components = (%.3f, %.3f, %.3f)" % (*p, *e))
    report.check_at_most("map: mapped points and rotated e_theta", error,
                         ROUNDING_TOL)
    # The result says which basis it is in: that of the query points.
    report.check("map: the rotated values are tagged Cartesian",
                 rotated.get_basis() == vector_in(cartesian), True)

    # ---------------------------------------------------------------------
    # 2. A scalar field at Cartesian points
    #
    # The source is the box in (r, theta, z), with P1 spaces of one and of
    # three components. QueryPoints takes the map as its third argument: the
    # points given to locate are Cartesian, they are mapped to (r, theta, z),
    # located there, and the field is evaluated at the mapped points.
    # g = 1 + 2 r + 3 z is linear in the mesh's coordinates, so it is
    # reproduced exactly, though it is not linear in x and y.
    source_mesh = omega_h.build_box(lib.world(), omega_h.Family.SIMPLEX, 2.0,
                                    1.6, 1.1, 8, 8, 6, False)
    source_geometry = pcms.backend.omega_h.create_geometry(source_mesh,
                                                           cylindrical)
    backend = pcms.LagrangeBackend.OmegaH
    scalars = pcms.create_lagrange_space(source_geometry, 1, 1,
                                         backend=backend)
    vectors = pcms.create_lagrange_space(source_geometry, 1, 3,
                                         backend=backend)

    g = scalars.create_function("g")
    holders = g.get_layout().get_dof_holder_coordinates().get_values()
    g.set_dof_holder_data(
        (1.0 + 2.0 * holders[:, 0] + 3.0 * holders[:, 2]).reshape(-1, 1))

    # Points in the part of the cylinder the mesh covers: x, y > 0 keeps
    # theta between 0 and pi/2.
    random = np.random.RandomState(11)
    points = np.stack([random.uniform(0.2, 1.0, 100),
                       random.uniform(0.2, 1.0, 100),
                       random.uniform(0.1, 1.0, 100)], axis=1)
    to_cylindrical = pcms.CartesianToCylindrical()

    query = pcms.QueryPoints(source_geometry, pcms.OutOfBoundsPolicy(),
                             to_cylindrical)
    query.locate(pcms.CoordinateView(cartesian, points))
    scalar_out = np.zeros((len(points), 1))
    query.evaluate(g, scalar_out)
    r = np.hypot(points[:, 0], points[:, 1])
    exact = 1.0 + 2.0 * r + 3.0 * points[:, 2]
    report.check_at_most("scalar at Cartesian points: g(r(x, y), z)",
                         np.max(np.abs(scalar_out[:, 0] - exact)),
                         ROUNDING_TOL)

    # ---------------------------------------------------------------------
    # 3. A vector field, rotated into the basis of the query points
    #
    # B = r e_r + z e_z, stored in physical cylindrical components (r, 0, z)
    # at the DOF holders, is the vector (x, y, z) in Cartesian components.
    # Components are not the same in every system, so the field says which
    # basis its components are in: the default is that of the space, here the
    # cylindrical system of the mesh. The map knows how to turn them into
    # Cartesian components at each query point, and the output is declared to
    # be in that basis: asking for another one is an error, not a silent
    # mistake.
    b = vectors.create_function("B", pcms.values.Vector,
                                scaling=pcms.ComponentScaling.Physical)
    fill_vector(b, lambda p: np.stack(
        [p[:, 0], np.zeros(len(p)), p[:, 2]], axis=1))
    vector_out = np.zeros((len(points), 3))
    query.evaluate(b, pcms.ValueView(vector_in(cartesian), vector_out))
    report.check_at_most("vector at Cartesian points: B = (x, y, z)",
                         np.max(np.abs(vector_out - points)), ROUNDING_TOL)
    refused = False
    try:
        query.evaluate(b, pcms.ValueView(vector_in(cylindrical), vector_out))
    except RuntimeError:
        refused = True
    report.check("vector: an output in another basis is refused", refused,
                 refused)

    # ---------------------------------------------------------------------
    # 4. The same through an Interpolator
    #
    # To fill a whole field of another mesh with the values of this one, bind
    # the DOF holders of the target once with Geometry.bind and the map, and
    # give the bound points to an Interpolator, which applies to as many
    # fields as you like without locating again. The target is the unit cube
    # in Cartesian coordinates, which lies inside the source box.
    target_mesh = omega_h.build_box(lib.world(), omega_h.Family.SIMPLEX, 1.0,
                                    1.0, 1.0, 4, 4, 4, False)
    target = pcms.create_lagrange_space(
        pcms.backend.omega_h.create_geometry(target_mesh, cartesian), 1, 3,
        backend=backend)
    bound_target = source_geometry.bind(target, map=to_cylindrical)
    interpolate = pcms.Interpolator(vectors, target, bound_target)

    b_target = target.create_function("B", pcms.values.Vector)
    interpolate.apply(b, b_target)
    target_points = b_target.get_layout().get_dof_holder_coordinates() \
        .get_values()
    report.check_at_most(
        "interpolated onto a Cartesian mesh: B = (x, y, z)",
        np.max(np.abs(b_target.get_dof_holder_data().get_values()
                      - target_points)), ROUNDING_TOL)

    # ---------------------------------------------------------------------
    # 5. Cylindrical points, Cartesian components
    #
    # The basis of the values does not have to be that of the points. A field
    # can have its points in (r, theta, z) and its components in Cartesian
    # x, y, z: say so when creating the function, by giving the system of the
    # components. B = (x, y, z) = (r cos theta, r sin theta, z) is then the
    # data. The values are already in the basis of the target, so the map
    # leaves them alone and only maps the points. The field is not linear in
    # (r, theta, z), so the P1 field only approximates it.
    b_borrowed = vectors.create_function("B", pcms.values.Vector,
                                         system=cartesian)
    fill_vector(b_borrowed, lambda p: np.stack(
        [p[:, 0] * np.cos(p[:, 1]), p[:, 0] * np.sin(p[:, 1]), p[:, 2]],
        axis=1))
    b_borrowed_target = target.create_function("B", pcms.values.Vector)
    interpolate.apply(b_borrowed, b_borrowed_target)
    borrowed_error = np.max(np.abs(
        b_borrowed_target.get_dof_holder_data().get_values() - target_points))
    print("Cartesian components on cylindrical points: error %.3e"
          % borrowed_error)
    report.check_at_most("borrowed basis: B = (x, y, z) up to interpolation",
                         borrowed_error, BORROWED_BASIS_TOL)

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
