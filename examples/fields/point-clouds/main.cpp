// point-clouds: reconstructing a field from scattered samples.
//
// Not all data lives on a mesh or a grid: measurements, particles and the
// output of other codes are often just values at scattered points, the sites.
// PCMS reconstructs a field from them with moving least squares (MLS): the
// value at a point is a local polynomial fit to the sites around it, the
// point's supports, weighted by their distance.
//
// Two things control the fit and are separate in PCMS:
//   * which sites are the supports of a point, and how many: the point-cloud
//     geometry's NeighbourhoodOptions (radius, minimum number of supports, and
//     whether the radius may grow to reach the minimum);
//   * the fit itself: MLSOptions (polynomial degree, weight function).
//
// What you will see, in order:
//   1. 400 random sites and a known function, a reconstruction space on them,
//      and the reconstruction evaluated at new points and onto a grid;
//   2. the radius and its adaptation: a radius that is too small for the
//      sites around a point, with and without adaptation;
//   3. sites that are the vertices or the cell centroids of a mesh, where the
//      supports come from the mesh's adjacency instead of a distance search.

#include <pcms/backends/point_cloud/geometry.hpp>
#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/configuration.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/localization/mls_options.hpp>
#include <pcms/transfer/interpolator.hpp>
#include <pcms/utility/assert.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>

#include <Kokkos_Core.hpp>
#ifdef PCMS_ENABLE_OMEGA_H
#include <pcms/backends/omega_h/omega_h.hpp>

#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#endif

#include "example_utils.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>

namespace
{
using pcms::Real;

// The function that was sampled: smooth, not a polynomial.
Real Sampled(Real x, Real y)
{
  return std::sin(2.0 * x) * std::cos(1.5 * y) + 0.5 * x * y;
}
Real Linear(Real x, Real y)
{
  return 1.0 + 2.0 * x - 3.0 * y;
}

// Quadratic MLS on 400 sites, spacing about 0.05: the error of a local
// quadratic fit is of order h^3 times the third derivatives of the function
// (about 8), here measured at 7e-4; the bounds leave a factor of three.
constexpr Real GridTol = 2e-3;
constexpr Real PointsTol = 2e-3;
// A linear fit reproduces a linear function whatever the supports are, up to
// the truncation of small singular values (MLSOptions::tol).
constexpr Real LinearTol = 1e-5;

// Sets the values of a field at its DOF holders (the sites) from f.
template <typename F>
void SetFromFunction(pcms::Field<Real>& field, F f)
{
  const auto holders = example::DofCoordinates(field);
  example::HostMatrix values("values", holders.extent(0), 1);
  for (std::size_t i = 0; i < holders.extent(0); ++i) {
    values(i, 0) = f(holders(i, 0), holders(i, 1));
  }
  example::SetDofData(field, values);
}

// The largest difference between the field at the points and f.
template <typename F>
Real MaxError(const pcms::Function<Real>& field,
              const std::shared_ptr<const pcms::CoordinateSystem>& system,
              const example::HostMatrix& points, F f)
{
  pcms::QueryPoints query(field.GetSpace().GetGeometry());
  query.Locate(example::Points(system, example::ToDevice(points)));
  example::DeviceMatrix out("out", points.extent(0), 1);
  query.Evaluate(field, example::Output(field.GetData().GetValueBasis(), out));
  const auto result = example::ToHost(out);
  Real error = 0.0;
  for (std::size_t i = 0; i < points.extent(0); ++i) {
    error =
      std::fmax(error, std::fabs(result(i, 0) - f(points(i, 0), points(i, 1))));
  }
  return error;
}

// Random points of the square [low, high]^2.
example::HostMatrix RandomPoints(std::mt19937& random, std::size_t n, Real low,
                                 Real high)
{
  std::uniform_real_distribution<Real> coordinate(low, high);
  example::HostMatrix points("points", n, 2);
  for (std::size_t i = 0; i < n; ++i) {
    points(i, 0) = coordinate(random);
    points(i, 1) = coordinate(random);
  }
  return points;
}

// A point cloud over the rows of `sites`, with a reconstruction space of
// polynomial degree `degree` on it.
std::shared_ptr<const pcms::FunctionSpace> ReconstructionSpace(
  const std::shared_ptr<const pcms::CoordinateSystem>& system,
  const example::HostMatrix& sites, pcms::NeighbourhoodOptions neighbourhood,
  unsigned degree)
{
  const auto geometry = pcms::backend::point_cloud::CreatePointCloudGeometry(
    pcms::CoordinateView<pcms::HostMemorySpace>(
      system, pcms::MakeConstRank2View(sites)),
    neighbourhood);
  return pcms::CreatePolynomialReconstructionSpace(
    geometry, pcms::MLSOptions{.degree = degree});
}
} // namespace

int main(int argc, char** argv)
{
#ifdef PCMS_ENABLE_OMEGA_H
  Omega_h::Library lib(&argc, &argv);
#else
  Kokkos::ScopeGuard kokkos(argc, argv);
#endif
  example::Report report;
  {
    const auto cartesian = pcms::csys::Cartesian::Create(2);

    // -------------------------------------------------------------------
    // 1. Scattered sites, a known function, a reconstruction
    //
    // The sites are random points of the unit square. A point-cloud geometry
    // holds them (copied, so the array may go away), with the options of its
    // support search: a point's supports are the sites within `radius` of it,
    // and at least `min_supports` of them are needed for the fit. MLSOptions
    // chooses a local quadratic fit with the default Gaussian weight.
    //
    // The sites are the DOF holders of the space: a field of the space has one
    // value per site.
    std::mt19937 random(5);
    const auto sites = RandomPoints(random, 400, 0.0, 1.0);
    const pcms::NeighbourhoodOptions neighbourhood{
      .radius = 0.12, .min_supports = 15, .adapt_radius = true};
    const auto space = ReconstructionSpace(cartesian, sites, neighbourhood, 2);
    auto samples = space->CreateFunction<Real>("samples");
    SetFromFunction(samples, Sampled);
    std::printf("%d sites\n",
                static_cast<int>(samples.GetLayout().GetNumDofHolder()));

    // Reconstruction at new points: nothing is special about them, QueryPoints
    // works as for any other field. Points near the boundary of the cloud have
    // supports on one side only, so they stay away from it here.
    const auto points = RandomPoints(random, 200, 0.1, 0.9);
    report.CheckAtMost("reconstruction at new points",
                       MaxError(samples, cartesian, points, Sampled),
                       PointsTol);

    // Reconstruction onto a grid: an Interpolator evaluates a source at the
    // DOF holders of a target space, here a P1 space on a uniform grid, and
    // keeps the located points for the next field.
    const pcms::UniformGrid<2> grid{
      .edge_length = {0.8, 0.8}, .bot_left = {0.1, 0.1}, .divisions = {20, 20}};
    const auto grid_space = pcms::CreateLagrangeSpace(
      pcms::backend::uniform_grid::CreateUniformGridGeometry(grid, cartesian),
      1, 1);
    auto on_grid = grid_space->CreateFunction<Real>("on_grid");
    pcms::Interpolator<Real> to_grid(*space, *grid_space);
    to_grid.Apply(samples, on_grid);
    const auto grid_points = example::DofCoordinates(on_grid);
    const auto grid_values = example::DofData(on_grid);
    Real grid_error = 0.0;
    for (std::size_t i = 0; i < grid_points.extent(0); ++i) {
      grid_error = std::fmax(
        grid_error, std::fabs(grid_values(i, 0) -
                              Sampled(grid_points(i, 0), grid_points(i, 1))));
    }
    report.CheckAtMost("reconstruction onto a 21 x 21 grid", grid_error,
                       GridTol);

    // -------------------------------------------------------------------
    // 2. The radius and its adaptation
    //
    // 400 sites in the unit square are 0.05 apart on average. A support
    // radius of 0.03 is smaller than that, so many points have no site within
    // it, and the others have fewer than the 15 that min_supports asks for.
    // What happens is the choice of `adapt_radius`:
    //   false  the radius is what it is: a point with no site within it is
    //          outside the domain of the reconstruction, which the
    //          out-of-bounds policy handles (Locate throws, by default);
    //   true   the radius grows, for all points together, until every point
    //          has min_supports sites, and the reconstruction is as good as
    //          with a radius that was right to begin with.
    pcms::NeighbourhoodOptions small = neighbourhood;
    small.radius = 0.03;

    small.adapt_radius = false;
    bool refused = false;
    try {
      const auto strict = ReconstructionSpace(cartesian, sites, small, 2);
      auto strict_samples = strict->CreateFunction<Real>("samples");
      SetFromFunction(strict_samples, Sampled);
      (void)MaxError(strict_samples, cartesian, points, Sampled);
    } catch (const pcms::pcms_error& e) {
      refused = true;
      std::printf("fixed radius 0.03: %s\n", e.what());
    }
    report.Check("radius too small, no adaptation: refused", refused,
                 refused ? 1 : 0);

    small.adapt_radius = true;
    const auto adapted = ReconstructionSpace(cartesian, sites, small, 2);
    auto adapted_samples = adapted->CreateFunction<Real>("samples");
    SetFromFunction(adapted_samples, Sampled);
    const Real adapted_error =
      MaxError(adapted_samples, cartesian, points, Sampled);
    std::printf("radius 0.03 with adaptation: error %.3e\n", adapted_error);
    report.CheckAtMost("radius too small, with adaptation", adapted_error,
                       PointsTol);

#ifdef PCMS_ENABLE_OMEGA_H
    // -------------------------------------------------------------------
    // 3. The sites are the vertices or the cell centroids of a mesh
    //
    // When the sites are mesh entities, a distance search is not needed: the
    // mesh knows which entities are neighbours. The geometry of the mesh
    // gives a point-cloud geometry over its vertices (entity dimension 0) or
    // over the centroids of its cells (the dimension of the mesh), and the
    // supports of a point are found by walking the mesh's adjacency from the
    // cell the point is in. Spaces made on the same mesh geometry have the
    // same entities, so this is how data stored on cells is reconstructed at
    // vertices, or the other way round.
    //
    // The function is linear, so a linear fit reproduces it from any
    // supports: the check is that the right values arrive at the right
    // places.
    auto mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 8, 8, 0);
    const auto mesh_geometry =
      pcms::backend::omega_h::CreateGeometry(mesh, cartesian);
    const pcms::NeighbourhoodOptions mesh_neighbourhood{
      .radius = 0.3, .min_supports = 8, .adapt_radius = true};
    const pcms::MLSOptions linear_fit{.degree = 1};

    const auto from_centroids = pcms::CreatePolynomialReconstructionSpace(
      pcms::backend::omega_h::CreatePointCloudGeometry(
        mesh_geometry, /*entity_dim=*/2, mesh_neighbourhood),
      linear_fit);
    auto at_centroids = from_centroids->CreateFunction<Real>("centroids");
    SetFromFunction(at_centroids, Linear);
    const auto vertices = example::DofCoordinates(
      pcms::CreateLagrangeSpace(
        mesh_geometry, 1, 1,
        pcms::LagrangeOptions{.backend = pcms::LagrangeBackend::OmegaH})
        ->CreateFunction<Real>("vertices"));
    report.CheckAtMost("cell centroids reconstructed at the vertices",
                       MaxError(at_centroids, cartesian, vertices, Linear),
                       LinearTol);

    const auto from_vertices = pcms::CreatePolynomialReconstructionSpace(
      pcms::backend::omega_h::CreatePointCloudGeometry(
        mesh_geometry, /*entity_dim=*/0, mesh_neighbourhood),
      linear_fit);
    auto at_vertices = from_vertices->CreateFunction<Real>("vertices");
    SetFromFunction(at_vertices, Linear);
    report.CheckAtMost("vertices reconstructed at the cell centroids",
                       MaxError(at_vertices, cartesian,
                                example::DofCoordinates(at_centroids), Linear),
                       LinearTol);
#endif
  }
  return report.Finish();
}
