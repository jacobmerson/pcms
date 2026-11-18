#!/usr/bin/env python3
"""point-clouds: reconstructing a field from scattered samples.

Not all data lives on a mesh or a grid: measurements, particles and the output
of other codes are often just values at scattered points, the sites. PCMS
reconstructs a field from them with moving least squares (MLS): the value at a
point is a local polynomial fit to the sites around it, the point's supports,
weighted by their distance. This is the Python twin of main.cpp.

Two things control the fit and are separate in PCMS:
  * which sites are the supports of a point, and how many: the point-cloud
    geometry's NeighbourhoodOptions (radius, minimum number of supports, and
    whether the radius may grow to reach the minimum);
  * the fit itself: MLSOptions (polynomial degree, weight function).

  1. 400 random sites and a known function, a reconstruction space on them,
     and the reconstruction evaluated at new points and onto a grid;
  2. the radius and its adaptation: a radius that is too small for the sites
     around a point, with and without adaptation;
  3. sites that are the vertices or the cell centroids of a mesh, where the
     supports come from the mesh's adjacency instead of a distance search.
"""
import pathlib
import sys

import numpy as np
import pcms
import PyOmega_h as omega_h

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "common"))
from example_utils import Report  # noqa: E402

# Quadratic MLS on 400 sites, spacing about 0.05: the error of a local
# quadratic fit is of order h^3 times the third derivatives of the function
# (about 8), here measured at 7e-4; the bounds leave a factor of three.
GRID_TOL = 2e-3
POINTS_TOL = 2e-3
# A linear fit reproduces a linear function whatever the supports are, up to
# the truncation of small singular values (MLSOptions.tol).
LINEAR_TOL = 1e-5


def sampled(x, y):
    """The function that was sampled: smooth, not a polynomial."""
    return np.sin(2.0 * x) * np.cos(1.5 * y) + 0.5 * x * y


def linear(x, y):
    return 1.0 + 2.0 * x - 3.0 * y


def set_from_function(field, f):
    """Set the values of a field at its DOF holders (the sites) from f."""
    holders = field.get_layout().get_dof_holder_coordinates().get_values()
    field.set_dof_holder_data(f(holders[:, 0], holders[:, 1]).reshape(-1, 1))


def max_error(field, points, f):
    """The largest difference between the field at the points and f."""
    query = pcms.QueryPoints(field.get_space().get_geometry())
    query.locate(pcms.CoordinateView(points))
    out = np.zeros((len(points), 1))
    query.evaluate(field, out)
    return np.max(np.abs(out[:, 0] - f(points[:, 0], points[:, 1])))


def reconstruction_space(sites, radius, min_supports, adapt_radius, degree):
    """A point cloud over the rows of `sites`, with a reconstruction space of
    polynomial degree `degree` on it."""
    neighbourhood = pcms.NeighbourhoodOptions()
    neighbourhood.radius = radius
    neighbourhood.min_supports = min_supports
    neighbourhood.adapt_radius = adapt_radius
    options = pcms.MLSOptions()
    options.degree = degree
    return pcms.create_polynomial_reconstruction_space(
        pcms.create_point_cloud_geometry(pcms.CoordinateView(sites),
                                         neighbourhood), options)


def main():
    report = Report()

    # ---------------------------------------------------------------------
    # 1. Scattered sites, a known function, a reconstruction
    #
    # The sites are random points of the unit square. A point-cloud geometry
    # holds them (copied, so the array may go away), with the options of its
    # support search: a point's supports are the sites within `radius` of it,
    # and at least `min_supports` of them are needed for the fit. MLSOptions
    # chooses a local quadratic fit with the default Gaussian weight.
    #
    # The sites are the DOF holders of the space: a field of the space has one
    # value per site.
    random = np.random.RandomState(5)
    sites = random.uniform(0.0, 1.0, size=(400, 2))
    space = reconstruction_space(sites, 0.12, 15, True, 2)
    samples = space.create_function("samples")
    set_from_function(samples, sampled)
    print("%d sites" % samples.get_layout().get_num_dof_holder())

    # Reconstruction at new points: nothing is special about them,
    # QueryPoints works as for any other field. Points near the boundary of
    # the cloud have supports on one side only, so they stay away from it
    # here.
    points = random.uniform(0.1, 0.9, size=(200, 2))
    report.check_at_most("reconstruction at new points",
                         max_error(samples, points, sampled), POINTS_TOL)

    # Reconstruction onto a grid: an Interpolator evaluates a source at the
    # DOF holders of a target space, here a P1 space on a uniform grid, and
    # keeps the located points for the next field.
    grid = pcms.UniformGrid2D()
    grid.bot_left = [0.1, 0.1]
    grid.edge_length = [0.8, 0.8]
    grid.divisions = [20, 20]
    grid_space = pcms.create_lagrange_space(
        pcms.create_uniform_grid_geometry(grid), 1, 1)
    on_grid = grid_space.create_function("on_grid")
    pcms.Interpolator(space, grid_space).apply(samples, on_grid)
    grid_points = on_grid.get_layout().get_dof_holder_coordinates() \
        .get_values()
    report.check_at_most(
        "reconstruction onto a 21 x 21 grid",
        np.max(np.abs(on_grid.get_dof_holder_data().get_values()[:, 0]
                      - sampled(grid_points[:, 0], grid_points[:, 1]))),
        GRID_TOL)

    # ---------------------------------------------------------------------
    # 2. The radius and its adaptation
    #
    # 400 sites in the unit square are 0.05 apart on average. A support
    # radius of 0.03 is smaller than that, so many points have no site within
    # it, and the others have fewer than the 15 that min_supports asks for.
    # What happens is the choice of `adapt_radius`:
    #   false  the radius is what it is: a point with no site within it is
    #          outside the domain of the reconstruction, which the
    #          out-of-bounds policy handles (locate raises, by default);
    #   true   the radius grows, for all points together, until every point
    #          has min_supports sites, and the reconstruction is as good as
    #          with a radius that was right to begin with.
    refused = False
    try:
        strict = reconstruction_space(sites, 0.03, 15, False, 2)
        strict_samples = strict.create_function("samples")
        set_from_function(strict_samples, sampled)
        max_error(strict_samples, points, sampled)
    except RuntimeError as e:
        refused = True
        print("fixed radius 0.03: %s" % e)
    report.check("radius too small, no adaptation: refused", refused, refused)

    adapted = reconstruction_space(sites, 0.03, 15, True, 2)
    adapted_samples = adapted.create_function("samples")
    set_from_function(adapted_samples, sampled)
    adapted_error = max_error(adapted_samples, points, sampled)
    print("radius 0.03 with adaptation: error %.3e" % adapted_error)
    report.check_at_most("radius too small, with adaptation", adapted_error,
                         POINTS_TOL)

    # ---------------------------------------------------------------------
    # 3. The sites are the vertices or the cell centroids of a mesh
    #
    # When the sites are mesh entities, a distance search is not needed: the
    # mesh knows which entities are neighbours. The geometry of the mesh
    # gives a point-cloud geometry over its vertices (entity dimension 0) or
    # over the centroids of its cells (the dimension of the mesh), and the
    # supports of a point are found by walking the mesh's adjacency from the
    # cell the point is in. Spaces made on the same mesh geometry have the
    # same entities, so this is how data stored on cells is reconstructed at
    # vertices, or the other way round.
    #
    # The function is linear, so a linear fit reproduces it from any
    # supports: the check is that the right values arrive at the right
    # places.
    lib = omega_h.OmegaHLibrary()
    mesh = omega_h.build_box(lib.world(), omega_h.Family.SIMPLEX, 1.0, 1.0,
                             0.0, 8, 8, 0, False)
    mesh_geometry = pcms.backend.omega_h.create_geometry(mesh)
    neighbourhood = pcms.NeighbourhoodOptions()
    neighbourhood.radius = 0.3
    neighbourhood.min_supports = 8
    neighbourhood.adapt_radius = True
    linear_fit = pcms.MLSOptions()
    linear_fit.degree = 1

    from_centroids = pcms.create_polynomial_reconstruction_space(
        pcms.backend.omega_h.create_point_cloud_geometry(
            mesh_geometry, 2, neighbourhood), linear_fit)
    at_centroids = from_centroids.create_function("centroids")
    set_from_function(at_centroids, linear)
    vertex_space = pcms.create_lagrange_space(
        mesh_geometry, 1, 1, backend=pcms.LagrangeBackend.OmegaH)
    vertices = np.array(
        vertex_space.create_function("vertices").get_layout()
        .get_dof_holder_coordinates().get_values())
    report.check_at_most("cell centroids reconstructed at the vertices",
                         max_error(at_centroids, vertices, linear),
                         LINEAR_TOL)

    from_vertices = pcms.create_polynomial_reconstruction_space(
        pcms.backend.omega_h.create_point_cloud_geometry(
            mesh_geometry, 0, neighbourhood), linear_fit)
    at_vertices = from_vertices.create_function("vertices")
    set_from_function(at_vertices, linear)
    centroids = np.array(
        at_centroids.get_layout().get_dof_holder_coordinates().get_values())
    report.check_at_most("vertices reconstructed at the cell centroids",
                         max_error(at_vertices, centroids, linear),
                         LINEAR_TOL)

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
