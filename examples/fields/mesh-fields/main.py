#!/usr/bin/env python3
"""mesh-fields: Lagrange fields on an Omega_h mesh.

hello-field evaluated a field on a grid. Here the geometry is an unstructured
simplex mesh, built with Omega_h (PyOmega_h in Python), and we look at what a
field on it consists of: where its DOF holders are, which of them this process
owns, and what the space's order changes (P0, P1 and, with MeshFields, P2).
This is the Python twin of main.cpp.

  1. an Omega_h box mesh, its geometry, and P0 and P1 spaces on it;
  2. the layout of a field: owned DOF holders and global counts;
  3. fields from functions, evaluated at points: P1 reproduces a linear
     function exactly, P0 (one value per element) does not;
  4. which differential operators a space supports;
  5. P2, a quadratic space, through MeshFields when PCMS has it.
"""
import pathlib
import sys

import numpy as np
import pcms
import PyOmega_h as omega_h

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from example_utils import Report  # noqa: E402

ROUNDING_TOL = 1e-12
GRADIENT_NORM = np.hypot(2.0, 3.0)


def linear(x, y):
    """f is linear: P1 contains it, P0 does not."""
    return 1.0 + 2.0 * x - 3.0 * y


def quadratic(x, y):
    """f is quadratic: P2 contains it, P1 does not."""
    return 1.0 + x * x + x * y - 2.0 * y * y


def set_from_function(field, f):
    """Set the DOF holders of `field` to f at their coordinates."""
    holders = field.get_layout().get_dof_holder_coordinates().get_values()
    field.set_dof_holder_data(f(holders[:, 0], holders[:, 1]).reshape(-1, 1))


def max_error(field, points, f):
    """The largest difference between the field at the points and f."""
    out = np.zeros((len(points), 1))
    pcms.evaluate(field, pcms.CoordinateView(points), out)
    return np.max(np.abs(out[:, 0] - f(points[:, 0], points[:, 1])))


def describe(name, layout, report, expected_global):
    """Print what the layout of a space says about its DOF holders."""
    print("%s: %d owned DOF holders of %d in all, distributed layout %s"
          % (name, layout.get_num_dof_holder(),
             layout.get_num_global_dof_holder(),
             "yes" if layout.is_distributed() else "no"))
    # On one process there are no ghosts: the owned DOF holders are all of
    # them, one per mesh entity.
    report.check(name + ": one DOF holder per entity",
                 layout.get_num_global_dof_holder() == expected_global
                 and layout.get_num_dof_holder() == expected_global,
                 layout.get_num_global_dof_holder())


def main():
    report = Report()

    # ---------------------------------------------------------------------
    # 1. Mesh, geometry and spaces
    #
    # A 2D box of 8 x 8 squares, each split into two triangles: 81 vertices
    # and 128 elements. The geometry of the mesh builds the search that finds
    # the element around a point; it is expensive, so one geometry is made
    # per mesh and shared by all spaces on the mesh.
    #
    # Two Lagrange spaces: P0 has one DOF holder per element (its centroid),
    # P1 one per vertex. LagrangeBackend.OmegaH selects the Omega_h
    # implementation, which has orders 0 and 1; the MeshFields backend has
    # orders 1 and 2 (see the end).
    lib = omega_h.OmegaHLibrary()
    mesh = omega_h.build_box(lib.world(), omega_h.Family.SIMPLEX, 1.0, 1.0,
                             0.0, 8, 8, 0, False)
    geometry = pcms.backend.omega_h.create_geometry(
        mesh, pcms.csys.Cartesian.create(2))
    backend = pcms.LagrangeBackend.OmegaH
    p0 = pcms.create_lagrange_space(geometry, 0, 1, backend=backend)
    p1 = pcms.create_lagrange_space(geometry, 1, 1, backend=backend)

    # ---------------------------------------------------------------------
    # 2. The layout of a field
    #
    # The layout of a space says how many DOF holders a field has and where
    # they are. When a mesh is partitioned over several processes, each
    # process holds the DOF holders of its part of the mesh. A DOF holder on
    # the boundary between two parts exists on both sides, but only one
    # process owns it: the other holds a ghost copy that Field.synchronize
    # refreshes from the owner. The numbers of a field are about the owned
    # DOF holders: get_num_dof_holder counts them, and the values you set and
    # get are those of the owned ones. get_num_global_dof_holder counts them
    # over all processes. (The per-holder ownership flags and global ids that
    # main.cpp prints are not in the Python API yet.)
    describe("P0 on elements", p0.get_layout(), report, mesh.nelems())
    describe("P1 on vertices", p1.get_layout(), report, mesh.nverts())

    # ---------------------------------------------------------------------
    # 3. Fields from functions, evaluated at points
    #
    # The same linear function in both spaces. evaluate gives the value of
    # the field at any point: for P1 that is the linear interpolation of the
    # vertex values of the element the point is in, which is exact for a
    # linear function. For P0 it is the value of the element, that is f at
    # the centroid: exact at the centroids, off by up to |grad f| times the
    # distance to the centroid elsewhere.
    f1 = p1.create_function("f1")
    f0 = p0.create_function("f0")
    set_from_function(f1, linear)
    set_from_function(f0, linear)

    points = np.random.RandomState(7).uniform(0.0, 1.0, size=(200, 2))
    report.check_at_most("P1: error of a linear field at random points",
                         max_error(f1, points, linear), ROUNDING_TOL)

    centroids = f0.get_layout().get_dof_holder_coordinates().get_values()
    report.check_at_most("P0: error at its own DOF holders (the centroids)",
                         max_error(f0, np.array(centroids), linear),
                         ROUNDING_TOL)
    element_size = np.sqrt(2.0) / 8.0  # longest edge
    p0_error = max_error(f0, points, linear)
    print("P0 error at random points: %.3e (the bound is %.3e)"
          % (p0_error, GRADIENT_NORM * element_size))
    report.check_at_most("P0: error at random points, at most |grad f| h",
                         p0_error, GRADIENT_NORM * element_size)

    # ---------------------------------------------------------------------
    # 4. What a space can compute
    #
    # A space says which differential operators its evaluators support.
    # Value is supported by all; the Lagrange spaces of this release do not
    # support Gradient yet (the spline space of splines-eqdsk does).
    for name, space in (("P0", p0), ("P1", p1)):
        print("%s supports Value: %s, Gradient: %s" % (
            name,
            "yes" if space.supports(pcms.DifferentialOperator.Value) else "no",
            "yes" if space.supports(pcms.DifferentialOperator.Gradient)
            else "no"))
    report.check("P1 supports evaluating values",
                 p1.supports(pcms.DifferentialOperator.Value), True)

    # ---------------------------------------------------------------------
    # 5. P2 through MeshFields
    #
    # Quadratic Lagrange fields have DOF holders at the vertices and at the
    # edge midpoints. They come from the MeshFields backend, which is
    # optional: creating the space raises a RuntimeError when PCMS was built
    # without it.
    try:
        p2 = pcms.create_lagrange_space(
            geometry, 2, 1, backend=pcms.LagrangeBackend.MeshFields)
    except RuntimeError as e:
        print("PCMS was built without MeshFields: P2 is skipped (%s)" % e)
    else:
        q2 = p2.create_function("q2")
        set_from_function(q2, quadratic)
        num_holders = q2.get_layout().get_num_dof_holder()
        print("P2: %d DOF holders (%d vertices and %d edges)"
              % (num_holders, mesh.nverts(), mesh.nedges()))
        report.check("P2: one DOF holder per vertex and per edge",
                     num_holders == mesh.nverts() + mesh.nedges(),
                     num_holders)
        report.check_at_most("P2: error of a quadratic field at random points",
                             max_error(q2, points, quadratic), 1e-10)
        # A linear field cannot represent the quadratic, so P1 is off.
        q1 = p1.create_function("q1")
        set_from_function(q1, quadratic)
        p1_error = max_error(q1, points, quadratic)
        print("P1 error for the quadratic function: %.3e" % p1_error)
        report.check("P1 cannot represent a quadratic function",
                     p1_error > 1e-3, True)

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
