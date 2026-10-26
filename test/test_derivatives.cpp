#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <pcms/field/coordinate_systems/cartesian.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/function_space/spline.h>
#include <pcms/field/gradient_source.hpp>
#include <pcms/transfer/interpolator.h>
#include "field_test_utils.h"
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pcms::DeviceMemorySpace;
using pcms::HostMemorySpace;
using pcms::Real;
using pcms::ValueBasis;
using pcms::ValueView;

namespace
{

// A bicubic not-a-knot spline reproduces this cubic exactly, and so its
// gradient.
KOKKOS_INLINE_FUNCTION Real Cubic(Real x, Real y)
{
  return x * x + 3.0 * x * y - y * y * y;
}

std::shared_ptr<pcms::SplineFunctionSpace> MakeSplineSpace()
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, -1.0};
  grid.edge_length = {2.0, 2.0};
  grid.divisions = {16, 16};
  return pcms::SplineFunctionSpace::FromUniformGrid(
    grid, pcms::csys::Cartesian::Create(2));
}

} // namespace

TEST_CASE("Spline gradients")
{
  auto space = MakeSplineSpace();
  auto f = space->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return Cubic(x, y); });

  const std::vector<Real> pts = {0.3, -0.7, 1.1, 0.2, 1.7, 0.9, 0.55, 0.45};
  const size_t n = pts.size() / 2;
  auto coords =
    pcms::test::CreateDeviceCoordinateView(pts, space->GetCoordinateSystem());
  const auto bound = space->Bind(coords.coordinate_view);
  auto values = space->CreatePointEvaluator<Real>(bound);
  auto gradient = space->CreatePointEvaluator<Real>(bound, 1);

  // The gradient is a physical covector on the space's coordinate system.
  const ValueBasis basis = gradient->OutputBasis(f.GetData().GetValueBasis());
  REQUIRE(basis.variance == pcms::values::Covector);
  REQUIRE(basis.scaling == pcms::ComponentScaling::Physical);
  REQUIRE(
    pcms::SameCoordinateSystem(basis.system, space->GetCoordinateSystem()));

  Kokkos::View<Real**, DeviceMemorySpace> grad_out("grad", n, 2);
  gradient->Evaluate(f, ValueView<Real, DeviceMemorySpace>(
                          basis, pcms::MakeRank2View(grad_out)));
  Kokkos::View<Real**, DeviceMemorySpace> value_out("values", n, 1);
  values->Evaluate(f, pcms::test::TagLike(f, value_out));
  const auto grad =
    Kokkos::create_mirror_view_and_copy(HostMemorySpace(), grad_out);
  const auto value =
    Kokkos::create_mirror_view_and_copy(HostMemorySpace(), value_out);
  for (size_t i = 0; i < n; ++i) {
    const Real x = pts[2 * i], y = pts[2 * i + 1];
    CAPTURE(i, x, y);
    REQUIRE_THAT(value(i, 0), WithinAbs(Cubic(x, y), pcms::test::EvalTol));
    REQUIRE_THAT(grad(i, 0), WithinAbs(2.0 * x + 3.0 * y, pcms::test::EvalTol));
    REQUIRE_THAT(grad(i, 1),
                 WithinAbs(3.0 * x - 3.0 * y * y, pcms::test::EvalTol));
  }

  // A gradient written into a view tagged as values is rejected.
  Kokkos::View<Real**, DeviceMemorySpace> wrong("wrong", n, 1);
  REQUIRE_THROWS_WITH(gradient->Evaluate(f, pcms::test::TagLike(f, wrong)),
                      ContainsSubstring("does not match the basis"));
  REQUIRE_THROWS_WITH(space->CreatePointEvaluator<Real>(bound, 2),
                      ContainsSubstring("derivative order 2"));
}

TEST_CASE("Gradients fill the out-of-bounds rows")
{
  auto space = MakeSplineSpace();
  auto f = space->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return Cubic(x, y); });
  const std::vector<Real> pts = {0.5, 0.5, 3.0, 0.0};
  auto coords =
    pcms::test::CreateDeviceCoordinateView(pts, space->GetCoordinateSystem());
  const auto bound =
    space->Bind(coords.coordinate_view,
                pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, -7.0});
  auto gradient = space->CreatePointEvaluator<Real>(bound, 1);
  Kokkos::View<Real**, DeviceMemorySpace> out("grad", 2, 2);
  gradient->Evaluate(f, ValueView<Real, DeviceMemorySpace>(
                          gradient->OutputBasis(f.GetData().GetValueBasis()),
                          pcms::MakeRank2View(out)));
  const auto grad = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), out);
  REQUIRE_THAT(grad(0, 0),
               WithinAbs(2.0 * 0.5 + 3.0 * 0.5, pcms::test::EvalTol));
  REQUIRE(grad(1, 0) == -7.0);
  REQUIRE(grad(1, 1) == -7.0);
}

TEST_CASE("Interpolating a field and its gradient over one localization")
{
  std::shared_ptr<const pcms::SplineFunctionSpace> source = MakeSplineSpace();
  auto f = source->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return Cubic(x, y); });

  // Targets on a coarser grid strictly inside the source grid.
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.2, -0.8};
  grid.edge_length = {1.5, 1.4};
  grid.divisions = {5, 7};
  auto value_space = pcms::LagrangeFunctionSpace::FromUniformGrid(
    grid, 1, source->GetCoordinateSystem());
  auto gradient_space = pcms::LagrangeFunctionSpace::FromUniformGrid(
    grid, 2, source->GetCoordinateSystem());
  auto value = value_space->CreateFunction<Real>("f");
  auto gradient =
    gradient_space->CreateFunction<Real>("grad f", pcms::values::Covector);

  const auto bound = source->Bind(*value_space);
  const auto grad_source = pcms::Gradient(source);
  pcms::Interpolator<Real> values(*source, *value_space, bound);
  pcms::Interpolator<Real> gradients(*grad_source, *gradient_space, bound);
  values.Apply(f, value);
  gradients.Apply(f, gradient);

  const auto coords = pcms::test::CopyCoordinatesToHost(
    value_space->GetLayout()->GetDOFHolderCoordinates().GetValues());
  const auto v = value.GetDOFHolderDataHost().GetValues();
  const auto g = gradient.GetDOFHolderDataHost().GetValues();
  for (size_t i = 0; i < coords.extent(0); ++i) {
    const Real x = coords(i, 0), y = coords(i, 1);
    CAPTURE(i, x, y);
    REQUIRE_THAT(v(i, 0), WithinAbs(Cubic(x, y), pcms::test::EvalTol));
    REQUIRE_THAT(g(i, 0), WithinAbs(2.0 * x + 3.0 * y, pcms::test::EvalTol));
    REQUIRE_THAT(g(i, 1),
                 WithinAbs(3.0 * x - 3.0 * y * y, pcms::test::EvalTol));
  }

  // A gradient source binds through the source, so it can also bind itself.
  pcms::Interpolator<Real> self_bound(*grad_source, *gradient_space);
  self_bound.Apply(f, gradient);
  REQUIRE_THAT(gradient.GetDOFHolderDataHost().GetValues()(0, 0),
               WithinAbs(g(0, 0), pcms::test::ExactTol));

  // Values are not written into a covector target.
  REQUIRE_THROWS_WITH(
    pcms::Interpolator<Real>(*source, *gradient_space, bound)
      .Apply(f, gradient),
    ContainsSubstring("differs from the target's declared basis"));
  // The bound points must be the target's DOF holders.
  pcms::UniformGrid<2> other_grid = grid;
  other_grid.divisions = {4, 7};
  auto other_space = pcms::LagrangeFunctionSpace::FromUniformGrid(
    other_grid, 1, source->GetCoordinateSystem());
  REQUIRE_THROWS_WITH(pcms::Interpolator<Real>(*source, *other_space, bound),
                      ContainsSubstring("target's DOF-holder sites"));
  // Second derivatives are not available from the spline yet.
  REQUIRE_THROWS_WITH(
    pcms::Gradient(grad_source)->CreatePointEvaluator<Real>(bound),
    ContainsSubstring("derivative order 2"));
}
