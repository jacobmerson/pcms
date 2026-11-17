#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <pcms/field/coordinate_systems/cartesian.hpp>
#include <pcms/field/evaluator/omega_h_geometry.hpp>
#include <pcms/field/evaluator/uniform_grid_geometry.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/function_space/spline.h>
#include <pcms/field/linear_combination.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/transfer/interpolator.h>
#include "field_test_utils.h"
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pcms::DeviceMemorySpace;
using pcms::DifferentialOperator;
using pcms::HostMemorySpace;
using pcms::OutOfBoundsMode;
using pcms::OutOfBoundsPolicy;
using pcms::Real;
using pcms::ValueBasis;

namespace
{

constexpr Real A = 2.0;
constexpr Real B = -0.5;

KOKKOS_INLINE_FUNCTION Real F(Real x, Real y)
{
  return x * x + 3.0 * x * y - y * y * y;
}

KOKKOS_INLINE_FUNCTION Real G(Real x, Real y)
{
  return 1.0 + x - 2.0 * y * y + x * y * y;
}

/// Evaluates `op` of `f` at the located points of `query` into a new
/// [NumPoints()][width] view tagged `basis`; returns a host copy.
auto EvaluateToHost(pcms::QueryPoints& query, const pcms::Function<Real>& f,
                    const ValueBasis& basis, int width = 1,
                    DifferentialOperator op = DifferentialOperator::Value)
{
  Kokkos::View<Real**, DeviceMemorySpace> out("out", query.Bound()->NumPoints(),
                                              width);
  query.Evaluate(
    f,
    pcms::ValueView<Real, DeviceMemorySpace>(basis, pcms::MakeRank2View(out)),
    op);
  Kokkos::View<Real**, HostMemorySpace> host("host", out.extent(0), width);
  pcms::DeepCopyMismatchLayouts(host, out);
  return host;
}

} // namespace

TEST_CASE("A LinearCombination of Lagrange functions evaluates as the "
          "combined function")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 8);
  const auto geometry =
    pcms::CreateOmegaHGeometry(mesh, pcms::csys::Cartesian::Deferred());
  // two spaces on one geometry: the terms need not share a space
  auto space = pcms::test::MakeP1Space(geometry);
  auto other_space = pcms::test::MakeP1Space(geometry);
  auto f = space->CreateFunction<Real>("f");
  auto g = other_space->CreateFunction<Real>("g");
  auto h = space->CreateFunction<Real>("h");
  pcms::test::SetField(f, OMEGA_H_LAMBDA(Real x, Real y) { return F(x, y); });
  pcms::test::SetField(g, OMEGA_H_LAMBDA(Real x, Real y) { return G(x, y); });
  pcms::test::SetField(
    h, OMEGA_H_LAMBDA(Real x, Real y) { return A * F(x, y) + B * G(x, y); });

  const auto combination =
    pcms::LinearCombination<Real>("A f + B g", {{A, f}, {B, g}});
  REQUIRE(combination.GetName() == "A f + B g");
  REQUIRE(combination.GetSpace().GetGeometry() == geometry);
  REQUIRE(pcms::SameValueBasis(combination.GetData().GetValueBasis(),
                               f.GetData().GetValueBasis()));
  REQUIRE(combination.GetData().GetDOFAccess() == nullptr);
  const auto basis = f.GetData().GetValueBasis();

  SECTION("QueryPoints evaluates it with the one bound source")
  {
    // the second point is outside the unit square
    const std::vector<Real> first{0.25, 0.5, 2.0, 2.0, 0.75, 0.1, 0.4, 0.9};
    const std::vector<Real> second{0.6, 0.6, 0.3, 0.2};
    auto first_coords = pcms::test::CreateDeviceCoordinateView(
      first, space->GetCoordinateSystem());
    auto second_coords = pcms::test::CreateDeviceCoordinateView(
      second, space->GetCoordinateSystem());
    pcms::QueryPoints query(geometry,
                            OutOfBoundsPolicy{OutOfBoundsMode::FILL, -7.0});
    query.Locate(first_coords.coordinate_view);
    auto values = EvaluateToHost(query, combination, basis);
    auto expected = EvaluateToHost(query, h, basis);
    for (int i = 0; i < 4; ++i) {
      CAPTURE(i);
      REQUIRE_THAT(values(i, 0),
                   WithinAbs(expected(i, 0), pcms::test::EvalTol));
    }
    REQUIRE(values(1, 0) == -7.0);

    // the term evaluators hold QueryPoints' bound source and nothing else:
    // released by Locate, it is rebound in place
    const auto* address = query.Bound().get();
    query.Locate(second_coords.coordinate_view);
    REQUIRE(query.Bound().get() == address);
    values = EvaluateToHost(query, combination, basis);
    expected = EvaluateToHost(query, h, basis);
    for (int i = 0; i < 2; ++i) {
      CAPTURE(i);
      REQUIRE_THAT(values(i, 0),
                   WithinAbs(expected(i, 0), pcms::test::EvalTol));
    }

    // the terms are referenced: values written to them are seen
    pcms::test::SetField(
      g, OMEGA_H_LAMBDA(Real x, Real y) { return 2.0 * G(x, y); });
    pcms::test::SetField(
      h, OMEGA_H_LAMBDA(Real x, Real y) {
        return A * F(x, y) + 2.0 * B * G(x, y);
      });
    values = EvaluateToHost(query, combination, basis);
    expected = EvaluateToHost(query, h, basis);
    for (int i = 0; i < 2; ++i) {
      CAPTURE(i);
      REQUIRE_THAT(values(i, 0),
                   WithinAbs(expected(i, 0), pcms::test::EvalTol));
    }
  }

  SECTION("an operator a term does not support is reported")
  {
    REQUIRE_FALSE(
      combination.GetSpace().Supports(DifferentialOperator::Gradient));
    const std::vector<Real> pts{0.25, 0.5};
    auto coords =
      pcms::test::CreateDeviceCoordinateView(pts, space->GetCoordinateSystem());
    pcms::QueryPoints query(geometry);
    query.Locate(coords.coordinate_view);
    REQUIRE_THROWS_WITH(EvaluateToHost(query, combination, basis, 2,
                                       DifferentialOperator::Gradient),
                        ContainsSubstring("Gradient is not supported"));
  }

  SECTION("an Interpolator from its space")
  {
    // the target is on another geometry of the same mesh: its DOF holders
    // are the vertices, where P1 interpolation reproduces the DOFs
    auto target = pcms::test::MakeP1Space(mesh);
    auto t = target->CreateFunction<Real>("t");
    pcms::Interpolator<Real> onto_mesh(combination.GetSpace(), *target);
    onto_mesh.Apply(combination, t);
    const auto got = t.GetDOFHolderDataHost().GetValues();
    const auto want = h.GetDOFHolderDataHost().GetValues();
    REQUIRE(got.extent(0) == want.extent(0));
    for (size_t i = 0; i < got.extent(0); ++i) {
      CAPTURE(i);
      REQUIRE_THAT(got(i, 0), WithinAbs(want(i, 0), pcms::test::EvalTol));
    }
  }
}

TEST_CASE("A LinearCombination forwards the gradient")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, -1.0};
  grid.edge_length = {2.0, 2.0};
  grid.divisions = {16, 16};
  auto space = pcms::CreateSplineSpace(
    pcms::CreateUniformGridGeometry(grid, pcms::csys::Cartesian::Create(2)));
  auto f = space->CreateFunction<Real>("f");
  auto g = space->CreateFunction<Real>("g");
  auto h = space->CreateFunction<Real>("h");
  pcms::test::SetField(f, OMEGA_H_LAMBDA(Real x, Real y) { return F(x, y); });
  pcms::test::SetField(g, OMEGA_H_LAMBDA(Real x, Real y) { return G(x, y); });
  pcms::test::SetField(
    h, OMEGA_H_LAMBDA(Real x, Real y) { return A * F(x, y) + B * G(x, y); });
  const auto combination = pcms::LinearCombination<Real>("c", {{A, f}, {B, g}});
  REQUIRE(combination.GetSpace().Supports(DifferentialOperator::Gradient));

  const std::vector<Real> pts{0.3, -0.7, 1.1, 0.2, 0.55, 0.45};
  auto coords =
    pcms::test::CreateDeviceCoordinateView(pts, space->GetCoordinateSystem());
  pcms::QueryPoints query(space->GetGeometry());
  query.Locate(coords.coordinate_view);
  const ValueBasis basis = f.GetData().GetValueBasis();
  const ValueBasis gradient_basis =
    pcms::GradientBasis(basis, space->GetCoordinateSystem());
  const auto values = EvaluateToHost(query, combination, basis);
  const auto expected = EvaluateToHost(query, h, basis);
  const auto gradient = EvaluateToHost(query, combination, gradient_basis, 2,
                                       DifferentialOperator::Gradient);
  const auto expected_gradient =
    EvaluateToHost(query, h, gradient_basis, 2, DifferentialOperator::Gradient);
  for (size_t i = 0; i < pts.size() / 2; ++i) {
    CAPTURE(i);
    REQUIRE_THAT(values(i, 0), WithinAbs(expected(i, 0), pcms::test::EvalTol));
    for (int d = 0; d < 2; ++d) {
      CAPTURE(d);
      REQUIRE_THAT(gradient(i, d),
                   WithinAbs(expected_gradient(i, d), pcms::test::EvalTol));
    }
  }
}

TEST_CASE("LinearCombination checks its terms")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 4);
  const auto geometry =
    pcms::CreateOmegaHGeometry(mesh, pcms::csys::Cartesian::Deferred());
  auto space = pcms::test::MakeP1Space(geometry);
  auto f = space->CreateFunction<Real>("f");

  REQUIRE_THROWS_WITH(pcms::LinearCombination<Real>("none", {}),
                      ContainsSubstring("at least one term"));

  // another geometry of the same mesh
  auto elsewhere = pcms::test::MakeP1Space(mesh)->CreateFunction<Real>("e");
  REQUIRE_THROWS_WITH(
    pcms::LinearCombination<Real>("c", {{1.0, f}, {1.0, elsewhere}}),
    ContainsSubstring("term 1 ('e') is on another geometry"));

  // two components against one
  const auto two = pcms::CreateLagrangeSpace(
    geometry, 1, 2,
    pcms::LagrangeOptions{.backend = pcms::LagrangeBackend::OmegaH});
  auto pair = two->CreateFunction<Real>("pair", pcms::values::Vector);
  REQUIRE_THROWS_WITH(
    pcms::LinearCombination<Real>("c", {{1.0, f}, {1.0, pair}}),
    ContainsSubstring("has 2 components but term 0 has 1"));

  // a vector in physical and a covector in natural components: the same
  // width, other semantics
  auto covector = two->CreateFunction<Real>("covector", pcms::values::Covector,
                                            pcms::ComponentScaling::Natural);
  REQUIRE_THROWS_WITH(
    pcms::LinearCombination<Real>("c", {{1.0, pair}, {1.0, covector}}),
    ContainsSubstring("another value basis"));
}
