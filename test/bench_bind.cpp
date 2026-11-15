// Benchmarks of binding and evaluation. Hidden behind the [!benchmark] tag, so
// ctest does not run them; run them with
//   unit_tests "[!benchmark]" --benchmark-samples 20
// on one thread (OMP_NUM_THREADS=1).

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <pcms/configuration.h>
#include <pcms/field/coordinate_systems/cartesian.hpp>
#include <pcms/field/evaluator/omega_h_geometry.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/function_space/spline.h>
#include <pcms/transfer/interpolator.h>
#include "field_test_utils.h"
#include <memory>
#include <random>
#include <vector>

using pcms::DeviceMemorySpace;
using pcms::LagrangeFunctionSpace;
using pcms::Real;

namespace
{

constexpr int NumPoints = 100000;
constexpr unsigned Seed = 20261115;

/// NumPoints random points strictly inside the unit square, interleaved.
std::vector<Real> RandomPointsInUnitSquare()
{
  std::mt19937 generator(Seed);
  std::uniform_real_distribution<Real> coordinate(1e-6, 1.0 - 1e-6);
  std::vector<Real> points(2 * static_cast<size_t>(NumPoints));
  for (auto& x : points) {
    x = coordinate(generator);
  }
  return points;
}

Omega_h::Mesh BuildSquareMesh(Omega_h::Library& lib, int n)
{
  return Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, n, n,
                            0);
}

pcms::UniformGrid<2> UnitSquareGrid(int n)
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {1.0, 1.0};
  grid.divisions = {n, n};
  return grid;
}

/// Evaluates `field` into `out` and returns the first value.
Real EvaluateInto(const pcms::PointEvaluator<Real>& evaluator,
                  const pcms::Field<Real>& field, const pcms::ValueBasis& basis,
                  Kokkos::View<Real**, DeviceMemorySpace> out)
{
  evaluator.Evaluate(field, pcms::ValueView<Real, DeviceMemorySpace>(
                              basis, pcms::MakeRank2View(out)));
  Kokkos::fence();
  Real first = 0;
  Kokkos::deep_copy(first, Kokkos::subview(out, 0, 0));
  return first;
}

/// Benchmarks Bind and Evaluate of a linear scalar field on `space`, and
/// CreatePointEvaluator if `time_create_evaluator`.
void BenchmarkBindAndEvaluate(
  const std::string& name, const std::shared_ptr<LagrangeFunctionSpace>& space,
  const std::vector<Real>& points, bool time_create_evaluator = false)
{
  auto f = space->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });
  auto coords = pcms::test::CreateDeviceCoordinateView(
    points, space->GetCoordinateSystem());

  BENCHMARK(name + ": bind")
  {
    return space->GetGeometry()->Bind(coords.coordinate_view);
  };

  const auto bound = space->GetGeometry()->Bind(coords.coordinate_view);
  const auto evaluator = space->CreatePointEvaluator<Real>(bound);
  if (time_create_evaluator) {
    BENCHMARK(name + ": create evaluator")
    {
      return space->CreatePointEvaluator<Real>(bound);
    };
  }
  Kokkos::View<Real**, DeviceMemorySpace> out("out", NumPoints, 1);
  const auto basis = f.GetData().GetValueBasis();
  BENCHMARK(name + ": evaluate")
  {
    return EvaluateInto(*evaluator, f, basis, out);
  };
}

} // namespace

TEST_CASE("Bind benchmarks: Omega_h P1", "[!benchmark]")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 64);
  const auto cart = pcms::csys::Cartesian::Create(2);
  const auto points = RandomPointsInUnitSquare();

  BENCHMARK("omega_h P1: build space")
  {
    return LagrangeFunctionSpace::FromGeometry(
      pcms::OmegaHGeometry::Create(mesh, cart), 1, 1, "global",
      LagrangeFunctionSpace::Backend::OmegaH);
  };

  auto space = LagrangeFunctionSpace::FromGeometry(
    pcms::OmegaHGeometry::Create(mesh, cart), 1, 1, "global",
    LagrangeFunctionSpace::Backend::OmegaH);
  BenchmarkBindAndEvaluate("omega_h P1", space, points);

  auto coords = pcms::test::CreateDeviceCoordinateView(points, cart);
  BENCHMARK("omega_h P1 + P1x3: two binds")
  {
    auto scalars = LagrangeFunctionSpace::FromGeometry(
      pcms::OmegaHGeometry::Create(mesh, cart), 1, 1, "global",
      LagrangeFunctionSpace::Backend::OmegaH);
    auto vectors = LagrangeFunctionSpace::FromGeometry(
      pcms::OmegaHGeometry::Create(mesh, cart), 1, 3, "global",
      LagrangeFunctionSpace::Backend::OmegaH);
    auto scalar_bound = scalars->GetGeometry()->Bind(coords.coordinate_view);
    auto vector_bound = vectors->GetGeometry()->Bind(coords.coordinate_view);
    return scalar_bound->FoundPoints().extent(0) +
           vector_bound->FoundPoints().extent(0);
  };

  BENCHMARK("omega_h P1 + P1x3: one geometry, one bind")
  {
    auto geometry = pcms::OmegaHGeometry::Create(mesh, cart);
    auto scalars = LagrangeFunctionSpace::FromGeometry(
      geometry, 1, 1, "global", LagrangeFunctionSpace::Backend::OmegaH);
    auto vectors = LagrangeFunctionSpace::FromGeometry(
      geometry, 1, 3, "global", LagrangeFunctionSpace::Backend::OmegaH);
    // one bound source serves both spaces
    auto bound = geometry->Bind(coords.coordinate_view);
    return scalars->GetLayout()->GetNumComponents() +
           vectors->GetLayout()->GetNumComponents() +
           bound->FoundPoints().extent(0);
  };
}

#ifdef PCMS_ENABLE_MESHFIELDS
TEST_CASE("Bind benchmarks: MeshFields P1", "[!benchmark]")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 64);
  auto geometry =
    pcms::OmegaHGeometry::Create(mesh, pcms::csys::Cartesian::Create(2));
  auto space = LagrangeFunctionSpace::FromGeometry(
    geometry, 1, 1, "global", LagrangeFunctionSpace::Backend::MeshFields);
  BenchmarkBindAndEvaluate("meshfields P1", space, RandomPointsInUnitSquare(),
                           /*time_create_evaluator=*/true);
}
#endif

TEST_CASE("Bind benchmarks: uniform grid", "[!benchmark]")
{
  auto space = LagrangeFunctionSpace::FromUniformGrid(
    UnitSquareGrid(256), 1, pcms::csys::Cartesian::Create(2));
  BenchmarkBindAndEvaluate("uniform grid", space, RandomPointsInUnitSquare());
}

TEST_CASE("Bind benchmarks: spline", "[!benchmark]")
{
  auto space = pcms::SplineFunctionSpace::FromUniformGrid(
    UnitSquareGrid(64), pcms::csys::Cartesian::Create(2));
  auto f = space->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return x * x + 3.0 * x * y; });
  auto coords = pcms::test::CreateDeviceCoordinateView(
    RandomPointsInUnitSquare(), space->GetCoordinateSystem());

  BENCHMARK("spline: bind")
  {
    return space->GetGeometry()->Bind(coords.coordinate_view);
  };

  const auto bound = space->GetGeometry()->Bind(coords.coordinate_view);
  const auto values = space->CreatePointEvaluator<Real>(bound);
  const auto gradient = space->CreatePointEvaluator<Real>(
    bound, pcms::DifferentialOperator::Gradient);
  const auto value_basis = f.GetData().GetValueBasis();
  const auto gradient_basis = gradient->OutputBasis(value_basis);
  Kokkos::View<Real**, DeviceMemorySpace> value_out("value", NumPoints, 1);
  Kokkos::View<Real**, DeviceMemorySpace> gradient_out("gradient", NumPoints,
                                                       2);
  BENCHMARK("spline: value")
  {
    return EvaluateInto(*values, f, value_basis, value_out);
  };
  BENCHMARK("spline: gradient")
  {
    return EvaluateInto(*gradient, f, gradient_basis, gradient_out);
  };
}

TEST_CASE("Bind benchmarks: interpolator", "[!benchmark]")
{
  auto lib = Omega_h::Library{};
  auto source_mesh = BuildSquareMesh(lib, 64);
  auto target_mesh = BuildSquareMesh(lib, 48);
  const auto cart = pcms::csys::Cartesian::Create(2);
  auto source = LagrangeFunctionSpace::FromGeometry(
    pcms::OmegaHGeometry::Create(source_mesh, cart), 1, 1, "global",
    LagrangeFunctionSpace::Backend::OmegaH);
  auto target = LagrangeFunctionSpace::FromGeometry(
    pcms::OmegaHGeometry::Create(target_mesh, cart), 1, 1, "global",
    LagrangeFunctionSpace::Backend::OmegaH);

  BENCHMARK("interpolator: construct")
  {
    return std::make_unique<pcms::Interpolator<Real>>(*source, *target);
  };
}
