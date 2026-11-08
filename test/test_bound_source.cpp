#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <pcms/field/bound_source.hpp>
#include <pcms/field/coordinate_systems/cylindrical.hpp>
#include <pcms/field/function_space/lagrange.h>
#include "field_test_utils.h"
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pcms::DeviceMemorySpace;
using pcms::HostMemorySpace;
using pcms::OutOfBoundsMode;
using pcms::OutOfBoundsPolicy;
using pcms::Real;

namespace
{

Omega_h::Mesh BuildSquare(Omega_h::Library& lib)
{
  return Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 8, 8, 0,
                            false);
}

std::vector<Real> EvaluateHost(const pcms::PointEvaluator<Real>& evaluator,
                               const pcms::Field<Real>& field, size_t n)
{
  Kokkos::View<Real**, DeviceMemorySpace> out("out", n, 1);
  evaluator.Evaluate(field, pcms::test::TagLike(field, out));
  auto host = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), out);
  std::vector<Real> values(n);
  for (size_t i = 0; i < n; ++i) {
    values[i] = host(i, 0);
  }
  return values;
}

} // namespace

TEST_CASE("Evaluators created from one bound source share its points")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquare(lib);
  auto space = pcms::test::MakeP1Space(mesh);
  auto field = space->CreateFunction<Real>("f");
  pcms::test::SetField(
    field,
    OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });

  const auto pts = pcms::test::StandardEvalCoords2D();
  const size_t n = pts.size() / 2;
  auto bound = [&] {
    // The bound source keeps what it needs, so the caller's points may go.
    auto coords =
      pcms::test::CreateDeviceCoordinateView(pts, space->GetCoordinateSystem());
    return space->Bind(coords.coordinate_view);
  }();
  REQUIRE(bound->NumPoints() == static_cast<pcms::LO>(n));
  REQUIRE(bound->QueryLayout() == nullptr);
  REQUIRE(bound->Policy().mode == OutOfBoundsMode::ERROR);

  auto first = space->CreatePointEvaluator<Real>(bound);
  auto second = space->CreatePointEvaluator<Real>(bound);
  const auto a = EvaluateHost(*first, field, n);
  const auto b = EvaluateHost(*second, field, n);
  for (size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    const Real expected = pcms::test::linear_f(pts[2 * i], pts[2 * i + 1]);
    REQUIRE_THAT(a[i], WithinAbs(expected, pcms::test::EvalTol));
    REQUIRE_THAT(b[i], WithinAbs(expected, pcms::test::EvalTol));
  }
}

TEST_CASE("Binding to a function space keeps its layout and the policy")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquare(lib);
  auto space = pcms::test::MakeP1Space(mesh);
  auto target_mesh = BuildSquare(lib);
  auto target = pcms::test::MakeP1Space(target_mesh);
  const OutOfBoundsPolicy policy{OutOfBoundsMode::FILL, -1.0};

  auto bound = space->Bind(*target, policy);
  REQUIRE(bound->QueryLayout() == target->GetLayout().get());
  REQUIRE(
    bound->NumPoints() ==
    static_cast<pcms::LO>(
      target->GetLayout()->GetDOFHolderCoordinates().GetValues().extent(0)));
  REQUIRE(bound->Policy().mode == OutOfBoundsMode::FILL);
  REQUIRE(bound->Policy().fill_value == -1.0);
}

TEST_CASE("A bound source is accepted only by the factory that bound it")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquare(lib);
  auto space = pcms::test::MakeP1Space(mesh);
  auto other = pcms::test::MakeP1Space(mesh);
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pcms::test::StandardEvalCoords2D(), space->GetCoordinateSystem());
  auto bound = space->Bind(coords.coordinate_view);

  REQUIRE_THROWS_WITH(other->CreatePointEvaluator<Real>(bound),
                      ContainsSubstring("bound by another factory"));
  REQUIRE_THROWS_WITH(space->CreatePointEvaluator<Real>(nullptr),
                      ContainsSubstring("must not be null"));
  REQUIRE_THROWS_WITH(space->CreatePointEvaluator<Real>(bound, -1),
                      ContainsSubstring("non-negative"));
}

TEST_CASE("Binding checks the query coordinate system")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquare(lib);
  auto space = pcms::test::MakeP1Space(mesh);
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pcms::test::StandardEvalCoords2D(), pcms::csys::CylindricalRZ::Create());
  REQUIRE_THROWS_WITH(space->Bind(coords.coordinate_view),
                      ContainsSubstring("is not the factory's coordinate"));
}

TEST_CASE("Sources without derivatives reject a derivative order")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquare(lib);
  auto space = pcms::test::MakeP1Space(mesh);
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pcms::test::StandardEvalCoords2D(), space->GetCoordinateSystem());
  auto bound = space->Bind(coords.coordinate_view);
  REQUIRE_THROWS_WITH(space->CreatePointEvaluator<Real>(bound, 1),
                      ContainsSubstring("derivative"));
}
