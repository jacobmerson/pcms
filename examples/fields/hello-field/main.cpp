// hello-field: a field on a uniform grid, evaluated at points.
//
// This is the smallest PCMS program that does something: it builds a field,
// fills it, and asks for its value somewhere else. The other examples build
// on it. It needs only Kokkos and PCMS (no Omega_h, no MPI), so it is also
// what CI compiles against an installed PCMS to check the installation.
//
// What you will see, in order:
//   1. a geometry (a uniform grid) and a function space on it (P1 Lagrange);
//   2. a field of that space, with its values set from a function;
//   3. evaluation at points with QueryPoints, which keeps the work of
//      locating the points and redoes it only when the points change;
//   4. the same evaluation in one call, with Evaluate;
//   5. points outside the grid: an error, or a fill value.

#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/utility/assert.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>

#include <Kokkos_Core.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
// The function we put into the field. A linear function lies in the space of
// P1 Lagrange fields, so evaluating the field anywhere in the grid gives
// back exactly this function, up to rounding.
pcms::Real Linear(pcms::Real x, pcms::Real y)
{
  return 1.0 + 2.0 * x + 3.0 * y;
}

constexpr pcms::Real RoundingTol = 1e-12;

// The largest difference between row i of `result` and f at point i of
// `points` (x0, y0, x1, y1, ...).
template <typename F>
pcms::Real MaxError(const example::HostMatrix& result,
                    const std::vector<pcms::Real>& points, F f)
{
  pcms::Real error = 0.0;
  for (std::size_t i = 0; i < result.extent(0); ++i) {
    error = std::fmax(
      error, std::fabs(result(i, 0) - f(points[2 * i], points[2 * i + 1])));
  }
  return error;
}
} // namespace

int main(int argc, char** argv)
{
  // PCMS runs its kernels with Kokkos, which must be initialized first.
  Kokkos::ScopeGuard kokkos(argc, argv);
  example::Report report;

  // ---------------------------------------------------------------------
  // 1. Geometry and function space
  //
  // A Geometry says where things are: here a uniform grid of 8 x 8 cells on
  // the unit square, in Cartesian coordinates. A FunctionSpace is created on
  // a geometry and says what a field on it is made of: here linear (order 1)
  // Lagrange basis functions with one component, one degree of freedom (DOF)
  // at each grid vertex.
  const auto cartesian = pcms::csys::Cartesian::Create(2);
  const pcms::UniformGrid<2> grid{
    .edge_length = {1.0, 1.0}, .bot_left = {0.0, 0.0}, .divisions = {8, 8}};
  const auto geometry =
    pcms::backend::uniform_grid::CreateUniformGridGeometry(grid, cartesian);
  const auto space = pcms::CreateLagrangeSpace(geometry, /*order=*/1,
                                               /*num_components=*/1);

  // ---------------------------------------------------------------------
  // 2. A field, with values from a function
  //
  // A Function is a field (values) that knows its space, so it can be
  // evaluated. Its values live at the DOF holders, the places that hold the
  // degrees of freedom; the layout of the field tells where they are. We read
  // their coordinates, evaluate Linear there, and write the values back.
  auto source = space->CreateFunction<pcms::Real>("source");
  const auto holders = example::DofCoordinates(source);
  example::HostMatrix values("values", holders.extent(0), 1);
  for (std::size_t i = 0; i < holders.extent(0); ++i) {
    values(i, 0) = Linear(holders(i, 0), holders(i, 1));
  }
  example::SetDofData(source, values);
  std::printf("a P1 field with %d DOF holders on a %d x %d grid\n",
              static_cast<int>(source.GetLayout().GetNumDofHolder()),
              grid.divisions[0], grid.divisions[1]);

  // ---------------------------------------------------------------------
  // 3. Evaluation at points: QueryPoints
  //
  // Evaluating a field at arbitrary points takes two steps: finding out which
  // grid cell each point is in (locating), and combining the values of that
  // cell's DOFs (evaluating). QueryPoints does both and remembers the first,
  // so one set of points can be evaluated for several fields; give it new
  // points with Locate when the points move.
  const std::vector<pcms::Real> points{0.30, 0.40, 0.75, 0.10,
                                       0.60, 0.60, 0.05, 0.95};
  const int num_points = static_cast<int>(points.size() / 2);

  pcms::QueryPoints query(geometry);
  query.Locate(example::Points(cartesian, example::ToDevice(points, 2)));

  // The output has one row per point and one column per component, and is
  // tagged with the basis of the field's values (for a scalar, nothing to
  // say; for a vector the tag says which coordinate system the components
  // are in; see the coordinate-systems example).
  example::DeviceMatrix result("result", num_points, 1);
  query.Evaluate(source,
                 example::Output(source.GetData().GetValueBasis(), result));
  auto result_host = example::ToHost(result);
  report.CheckAtMost("QueryPoints: error of a linear field",
                     MaxError(result_host, points, Linear), RoundingTol);

  // A second field of the same space is evaluated at the same located
  // points without searching again.
  auto other = space->CreateFunction<pcms::Real>("other");
  for (std::size_t i = 0; i < holders.extent(0); ++i) {
    values(i, 0) = holders(i, 0) - holders(i, 1);
  }
  example::SetDofData(other, values);
  query.Evaluate(other,
                 example::Output(other.GetData().GetValueBasis(), result));
  result_host = example::ToHost(result);
  report.CheckAtMost("QueryPoints: a second field, same points",
                     MaxError(result_host, points,
                              [](pcms::Real x, pcms::Real y) { return x - y; }),
                     RoundingTol);

  // ---------------------------------------------------------------------
  // 4. The one-shot Evaluate
  //
  // When the points are used once, Evaluate does the locating and the
  // evaluating in one call. It is the same as creating a QueryPoints, calling
  // Locate and Evaluate once, and dropping it.
  pcms::Evaluate(source,
                 example::Points(cartesian, example::ToDevice(points, 2)),
                 example::Output(source.GetData().GetValueBasis(), result));
  result_host = example::ToHost(result);
  report.CheckAtMost("Evaluate: error of a linear field",
                     MaxError(result_host, points, Linear), RoundingTol);

  // ---------------------------------------------------------------------
  // 5. Points outside the grid
  //
  // The second and last points below are outside the unit square. What
  // happens is the choice of the OutOfBoundsPolicy, fixed when the points are
  // located:
  //   ERROR (the default)  Locate throws a pcms_error;
  //   FILL                 the points outside get a fill value, the others
  //                        are evaluated as usual; BoundSource::FoundPoints
  //                        and MissingPoints list which is which;
  //   NEAREST_BOUNDARY     the points are evaluated as if their nearest
  //                        grid cell extended past the boundary (only
  //                        uniform grids can do this).
  const std::vector<pcms::Real> mixed{0.50, 0.50, 1.50,  0.50,
                                      0.20, 0.80, -0.25, 0.30};
  const int num_mixed = static_cast<int>(mixed.size() / 2);
  const auto mixed_points = example::ToDevice(mixed, 2);

  bool threw = false;
  try {
    pcms::QueryPoints strict(geometry);
    strict.Locate(example::Points(cartesian, mixed_points));
  } catch (const pcms::pcms_error& e) {
    threw = true;
    std::printf("ERROR policy: %s\n", e.what());
  }
  report.Check("ERROR policy: a point outside throws", threw, threw ? 1 : 0);

  constexpr pcms::Real FillValue = -999.0;
  pcms::QueryPoints lenient(
    geometry, pcms::OutOfBoundsPolicy{.mode = pcms::OutOfBoundsMode::FILL,
                                      .fill_value = FillValue});
  lenient.Locate(example::Points(cartesian, mixed_points));
  example::DeviceMatrix filled("filled", num_mixed, 1);
  lenient.Evaluate(source,
                   example::Output(source.GetData().GetValueBasis(), filled));
  const auto filled_host = example::ToHost(filled);
  const auto missing = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), lenient.Bound()->MissingPoints());
  const auto found = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), lenient.Bound()->FoundPoints());
  std::printf("FILL policy: %d points found, %d missing:",
              static_cast<int>(found.extent(0)),
              static_cast<int>(missing.extent(0)));
  for (std::size_t k = 0; k < missing.extent(0); ++k) {
    std::printf(" #%d", missing(k));
  }
  std::printf("\n");
  report.Check("FILL policy: points outside the grid are missing",
               missing.extent(0) == 2, static_cast<long>(missing.extent(0)));
  pcms::Real error = 0.0;
  for (std::size_t k = 0; k < missing.extent(0); ++k) {
    error = std::fmax(error, std::fabs(filled_host(missing(k), 0) - FillValue));
  }
  for (std::size_t k = 0; k < found.extent(0); ++k) {
    const int i = found(k);
    error = std::fmax(error, std::fabs(filled_host(i, 0) -
                                       Linear(mixed[2 * i], mixed[2 * i + 1])));
  }
  report.CheckAtMost("FILL policy: fill value and field values", error,
                     RoundingTol);

  pcms::QueryPoints extended(
    geometry,
    pcms::OutOfBoundsPolicy{.mode = pcms::OutOfBoundsMode::NEAREST_BOUNDARY});
  extended.Locate(example::Points(cartesian, mixed_points));
  extended.Evaluate(source,
                    example::Output(source.GetData().GetValueBasis(), filled));
  report.CheckAtMost("NEAREST_BOUNDARY: a linear field extends exactly",
                     MaxError(example::ToHost(filled), mixed, Linear),
                     RoundingTol);

  return report.Finish();
}
