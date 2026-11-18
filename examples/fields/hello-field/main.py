#!/usr/bin/env python3
"""hello-field: a field on a uniform grid, evaluated at points.

The smallest PCMS program that does something: it builds a field, fills it,
and asks for its value somewhere else. This is the Python twin of main.cpp
and tells the same story with the same asserted result:

  1. a geometry (a uniform grid) and a function space on it (P1 Lagrange);
  2. a field of that space, with its values set from a function;
  3. evaluation at points with QueryPoints, which keeps the work of locating
     the points and redoes it only when the points change;
  4. the same evaluation in one call, with evaluate;
  5. points outside the grid: an error, or a fill value.

PCMS arrays are NumPy arrays: coordinates are (num_points, dim), values are
(num_points, num_components), always float64.
"""
import pathlib
import sys

import numpy as np
import pcms

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from example_utils import Report  # noqa: E402

ROUNDING_TOL = 1e-12


def linear(x, y):
    """The function we put into the field. A linear function lies in the
    space of P1 Lagrange fields, so evaluating the field anywhere in the grid
    gives back exactly this function, up to rounding."""
    return 1.0 + 2.0 * x + 3.0 * y


def max_error(result, points, f):
    """The largest difference between row i of `result` and f at point i."""
    return np.max(np.abs(result[:, 0] - f(points[:, 0], points[:, 1])))


def main():
    report = Report()

    # ---------------------------------------------------------------------
    # 1. Geometry and function space
    #
    # A geometry says where things are: here a uniform grid of 8 x 8 cells on
    # the unit square, in Cartesian coordinates (the default). A function
    # space is created on a geometry and says what a field on it is made of:
    # here linear (order 1) Lagrange basis functions with one component, one
    # degree of freedom (DOF) at each grid vertex.
    grid = pcms.UniformGrid2D()
    grid.bot_left = [0.0, 0.0]
    grid.edge_length = [1.0, 1.0]
    grid.divisions = [8, 8]
    geometry = pcms.create_uniform_grid_geometry(grid)
    space = pcms.create_lagrange_space(geometry, 1, 1)

    # ---------------------------------------------------------------------
    # 2. A field, with values from a function
    #
    # A Function is a field (values) that knows its space, so it can be
    # evaluated. Its values live at the DOF holders, the places that hold the
    # degrees of freedom; the layout of the field tells where they are. We
    # read their coordinates, evaluate linear there, and write the values
    # back.
    source = space.create_function("source")
    holders = source.get_layout().get_dof_holder_coordinates().get_values()
    source.set_dof_holder_data(
        linear(holders[:, 0], holders[:, 1]).reshape(-1, 1))
    print("a P1 field with %d DOF holders on a %d x %d grid"
          % (source.get_layout().get_num_dof_holder(), grid.divisions[0],
             grid.divisions[1]))

    # ---------------------------------------------------------------------
    # 3. Evaluation at points: QueryPoints
    #
    # Evaluating a field at arbitrary points takes two steps: finding out
    # which grid cell each point is in (locating), and combining the values of
    # that cell's DOFs (evaluating). QueryPoints does both and remembers the
    # first, so one set of points can be evaluated for several fields; give it
    # new points with locate when the points move.
    points = np.array([[0.30, 0.40], [0.75, 0.10], [0.60, 0.60],
                       [0.05, 0.95]])
    query = pcms.QueryPoints(geometry)
    query.locate(pcms.CoordinateView(points))

    # The output has one row per point and one column per component.
    result = np.zeros((len(points), 1))
    query.evaluate(source, result)
    report.check_at_most("QueryPoints: error of a linear field",
                         max_error(result, points, linear), ROUNDING_TOL)

    # A second field of the same space is evaluated at the same located
    # points without searching again.
    other = space.create_function("other")
    other.set_dof_holder_data((holders[:, 0] - holders[:, 1]).reshape(-1, 1))
    query.evaluate(other, result)
    report.check_at_most("QueryPoints: a second field, same points",
                         max_error(result, points, lambda x, y: x - y),
                         ROUNDING_TOL)

    # ---------------------------------------------------------------------
    # 4. The one-shot evaluate
    #
    # When the points are used once, evaluate does the locating and the
    # evaluating in one call. It is the same as creating a QueryPoints,
    # calling locate and evaluate once, and dropping it.
    pcms.evaluate(source, pcms.CoordinateView(points), result)
    report.check_at_most("evaluate: error of a linear field",
                         max_error(result, points, linear), ROUNDING_TOL)

    # ---------------------------------------------------------------------
    # 5. Points outside the grid
    #
    # The second and last points below are outside the unit square. What
    # happens is the choice of the OutOfBoundsPolicy, fixed when the points
    # are located:
    #   ERROR (the default)  locate raises a RuntimeError;
    #   FILL                 the points outside get a fill value, the others
    #                        are evaluated as usual; the bound source's
    #                        found_points and missing_points list which is
    #                        which;
    #   NEAREST_BOUNDARY     the points are evaluated as if their nearest
    #                        grid cell extended past the boundary (only
    #                        uniform grids can do this).
    mixed = np.array([[0.50, 0.50], [1.50, 0.50], [0.20, 0.80],
                      [-0.25, 0.30]])

    threw = False
    try:
        pcms.QueryPoints(geometry).locate(pcms.CoordinateView(mixed))
    except RuntimeError as e:
        threw = True
        print("ERROR policy: %s" % e)
    report.check("ERROR policy: a point outside throws", threw, threw)

    fill_value = -999.0
    lenient = pcms.QueryPoints(
        geometry, pcms.OutOfBoundsPolicy(pcms.OutOfBoundsMode.FILL,
                                         fill_value))
    lenient.locate(pcms.CoordinateView(mixed))
    filled = np.zeros((len(mixed), 1))
    lenient.evaluate(source, filled)
    found = lenient.bound.found_points()
    missing = lenient.bound.missing_points()
    print("FILL policy: %d points found, %d missing: %s"
          % (len(found), len(missing),
             " ".join("#%d" % i for i in missing)))
    report.check("FILL policy: points outside the grid are missing",
                 len(missing) == 2, len(missing))
    error = max(np.max(np.abs(filled[missing, 0] - fill_value)),
                max_error(filled[found], mixed[found], linear))
    report.check_at_most("FILL policy: fill value and field values", error,
                         ROUNDING_TOL)

    extended = pcms.QueryPoints(
        geometry,
        pcms.OutOfBoundsPolicy(pcms.OutOfBoundsMode.NEAREST_BOUNDARY))
    extended.locate(pcms.CoordinateView(mixed))
    extended.evaluate(source, filled)
    report.check_at_most("NEAREST_BOUNDARY: a linear field extends exactly",
                         max_error(filled, mixed, linear), ROUNDING_TOL)

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
