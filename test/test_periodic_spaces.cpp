#include "periodic_test_utils.hpp"
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <pcms/field/periodic_axis.hpp>
#include <pcms/transfer/forms.hpp>
#include <pcms/transfer/interpolator.hpp>
#include <pcms/transfer/l2_inner_product.hpp>
#include <pcms/transfer/projection.hpp>
#include <pcms/transfer/linear_solver.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <pcms/utility/coordinate_systems/parameter.hpp>
#include <algorithm>
#include <array>
#include <numbers>
#include <utility>

namespace
{

using pcms::Real;
using pcms::test::EvaluateAt;
using pcms::test::SetDofs;
constexpr Real TwoPi = 2 * std::numbers::pi_v<Real>;

std::shared_ptr<const pcms::Geometry> Periodic(int cells, Real period = TwoPi,
                                               Real origin = 0)
{
  return pcms::test::PeriodicGrid(cells, period, origin);
}

std::vector<std::vector<Real>> Points(int n, Real lo, Real hi)
{
  std::vector<std::vector<Real>> points;
  for (int i = 0; i < n; ++i) {
    points.push_back({lo + (hi - lo) * (i + 0.37) / n});
  }
  return points;
}

/// A 2D grid of `cells` cells per axis over x in [0, 1] and chi in [0, 2 pi),
/// periodic in chi, which is axis `chi_axis`.
std::shared_ptr<const pcms::Geometry> FluxGrid(int cells, int chi_axis)
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0, 0};
  grid.edge_length[1 - chi_axis] = 1;
  grid.edge_length[chi_axis] = TwoPi;
  grid.divisions = {cells, cells};
  return pcms::uniform_grid::CreateGeometry(
    grid, pcms::csys::CreateCartesian(2, {{chi_axis, 0.0, TwoPi}}));
}

/// The point (x, chi) with chi on axis `chi_axis`.
std::vector<Real> FluxPoint(Real x, Real chi, int chi_axis)
{
  return chi_axis == 1 ? std::vector<Real>{x, chi} : std::vector<Real>{chi, x};
}

/// f(x, chi) = x^3 + cos(chi) + x sin(2 chi) and its gradient in (x, chi).
std::array<Real, 3> Flux(Real x, Real chi)
{
  return {x * x * x + std::cos(chi) + x * std::sin(2 * chi),
          3 * x * x + std::sin(2 * chi),
          -std::sin(chi) + 2 * x * std::cos(2 * chi)};
}

/// The spline of Flux's node values on FluxGrid(cells, chi_axis).
std::pair<std::shared_ptr<const pcms::FunctionSpace>, pcms::Function<Real>>
FluxSpline(int cells, int chi_axis)
{
  const auto space = pcms::CreateSplineSpace(FluxGrid(cells, chi_axis));
  // the nodes, first axis fastest; chi has no last node
  const int along_first = chi_axis == 0 ? cells : cells + 1;
  std::vector<Real> values;
  for (int i = 0; i < space->GetLayout()->GetNumDofHolder(); ++i) {
    const int node[2] = {i % along_first, i / along_first};
    values.push_back(Flux(static_cast<Real>(node[1 - chi_axis]) / cells,
                          TwoPi * node[chi_axis] / cells)[0]);
  }
  auto f = space->CreateFunction<Real>();
  SetDofs(f, values);
  return {space, std::move(f)};
}

} // namespace

TEST_CASE("Fourier space: evaluation and derivative of a mode list with "
          "stride, including wrapped points",
          "[periodic][fourier]")
{
  const Real period = 4.0;
  const Real origin = -1.0;
  const auto geometry = Periodic(8, period, origin);
  const auto space = pcms::CreateFourierSpace(geometry, {0, 1, 3}, 2);
  // DOFs: 1; cos(2 t), sin(2 t); cos(6 t), sin(6 t)
  REQUIRE(space->GetLayout()->GetNumDofHolder() == 5);
  const std::vector<Real> c = {0.5, 2.0, -1.0, 0.25, 3.0};
  auto u = space->CreateFunction<Real>();
  SetDofs(u, c);
  const Real w = TwoPi / period;
  const auto f = [&](Real x) {
    const Real t = w * (x - origin);
    return c[0] + c[1] * std::cos(2 * t) + c[2] * std::sin(2 * t) +
           c[3] * std::cos(6 * t) + c[4] * std::sin(6 * t);
  };
  const auto df = [&](Real x) {
    const Real t = w * (x - origin);
    return w * (-2 * c[1] * std::sin(2 * t) + 2 * c[2] * std::cos(2 * t) -
                6 * c[3] * std::sin(6 * t) + 6 * c[4] * std::cos(6 * t));
  };
  // inside, below and above the interval
  const auto points = Points(24, -9.0, 11.0);
  const auto value = EvaluateAt(*space, u, points);
  const auto slope =
    EvaluateAt(*space, u, points, pcms::DifferentialOperator::Gradient);
  for (std::size_t p = 0; p < points.size(); ++p) {
    CHECK(value[p][0] == Catch::Approx(f(points[p][0])).margin(1e-11));
    CHECK(slope[p][0] == Catch::Approx(df(points[p][0])).margin(1e-10));
  }
  // the wrap: a point one period away has the same value
  const auto a = EvaluateAt(*space, u, {{0.3}});
  const auto b = EvaluateAt(*space, u, {{0.3 + period}});
  CHECK(a[0][0] == Catch::Approx(b[0][0]).margin(1e-11));
  CHECK_THROWS(pcms::CreateFourierSpace(geometry, {-1}));
  CHECK_THROWS(pcms::CreateFourierSpace(geometry, {1}, 0));
}

TEST_CASE("Periodic Lagrange order 1: nodes, midpoints and slopes",
          "[periodic][lagrange]")
{
  const int n = 6;
  const auto geometry = Periodic(n, 3.0, 1.0);
  const Real h = 0.5;
  const auto space = pcms::CreateLagrangeSpace(geometry, 1, 1);
  auto u = space->CreateFunction<Real>();
  const std::vector<Real> g = {1, 4, -2, 0.5, 3, 7};
  SetDofs(u, g);
  for (int i = 0; i < n; ++i) {
    const Real x = 1.0 + h * i;
    const Real next = g[(i + 1) % n];
    const auto v = EvaluateAt(*space, u, {{x}, {x + 0.5 * h}});
    CHECK(v[0][0] == Catch::Approx(g[i]).margin(1e-13));
    CHECK(v[1][0] == Catch::Approx(0.5 * (g[i] + next)).margin(1e-13));
    const auto d = EvaluateAt(*space, u, {{x + 0.25 * h}},
                              pcms::DifferentialOperator::Gradient);
    CHECK(d[0][0] == Catch::Approx((next - g[i]) / h).margin(1e-12));
  }
  // the last cell wraps to the first node
  const auto wrapped = EvaluateAt(*space, u, {{1.0 + 3.0 - 0.25 * h}});
  CHECK(wrapped[0][0] ==
        Catch::Approx(0.25 * g[5] + 0.75 * g[0]).margin(1e-12));
}

TEST_CASE("Periodic Lagrange order 1: an interpolator reaches target sites "
          "on the end of the period",
          "[periodic][lagrange]")
{
  const auto source = pcms::CreateLagrangeSpace(Periodic(4), 1, 1);
  pcms::UniformGrid<1> grid;
  grid.bot_left = {TwoPi / 2};
  grid.edge_length = {TwoPi / 2};
  grid.divisions = {2};
  const auto target = pcms::CreateLagrangeSpace(
    pcms::uniform_grid::CreateGeometry(
      grid, source->GetGeometry()->GetCoordinateSystem()),
    1, 1);
  auto u = source->CreateFunction<Real>();
  SetDofs(u, {1, 4, -2, 0.5});
  auto v = target->CreateFunction<Real>();
  pcms::CreateInterpolator<Real>(source, target)->Apply(u, v);
  // x = 2 pi is x = 0
  const auto values = pcms::test::GetDofs(v);
  REQUIRE(values.size() == 3);
  CHECK(values[0] == Catch::Approx(-2).margin(1e-13));
  CHECK(values[2] == Catch::Approx(1).margin(1e-13));
}

TEST_CASE("Periodic Hermite: cubics exactly away from the wrap, sine to "
          "fourth order",
          "[periodic][hermite]")
{
  const int n = 8;
  const Real period = 2.0;
  const Real h = period / n;
  const auto geometry = Periodic(n, period);
  const auto space = pcms::CreatePeriodicHermiteSpace(geometry);
  REQUIRE(space->GetLayout()->GetNumDofHolder() == 2 * n);
  // a cubic's values and derivatives at the nodes: the cells whose two nodes
  // are not across the wrap reproduce it
  const auto p = [](Real x) { return 1 + x - 2 * x * x + 0.5 * x * x * x; };
  const auto dp = [](Real x) { return 1 - 4 * x + 1.5 * x * x; };
  std::vector<Real> dofs;
  for (int i = 0; i < n; ++i) {
    dofs.push_back(p(h * i));
    dofs.push_back(dp(h * i));
  }
  auto u = space->CreateFunction<Real>();
  SetDofs(u, dofs);
  const auto points = Points(14, 0.0, period - h);
  const auto value = EvaluateAt(*space, u, points);
  const auto slope =
    EvaluateAt(*space, u, points, pcms::DifferentialOperator::Gradient);
  for (std::size_t q = 0; q < points.size(); ++q) {
    CHECK(value[q][0] == Catch::Approx(p(points[q][0])).margin(1e-12));
    CHECK(slope[q][0] == Catch::Approx(dp(points[q][0])).margin(1e-11));
  }
  // sin on the whole circle
  const auto error = [&](int cells) {
    const auto fine = pcms::CreatePeriodicHermiteSpace(Periodic(cells, TwoPi));
    std::vector<Real> d;
    for (int i = 0; i < cells; ++i) {
      d.push_back(std::sin(TwoPi * i / cells));
      d.push_back(std::cos(TwoPi * i / cells));
    }
    auto s = fine->CreateFunction<Real>();
    SetDofs(s, d);
    Real worst = 0;
    const auto pts = Points(50, -1.0, 8.0);
    const auto v = EvaluateAt(*fine, s, pts);
    for (std::size_t q = 0; q < pts.size(); ++q) {
      worst = std::max(worst, std::abs(v[q][0] - std::sin(pts[q][0])));
    }
    return worst;
  };
  const Real coarse = error(8);
  const Real fine = error(16);
  CHECK(coarse < 2e-3);
  CHECK(coarse / fine > 12.0); // fourth order: 16
}

TEST_CASE("Periodic cubic spline: interpolates the nodes, reproduces "
          "constants, approximates a sine",
          "[periodic][spline]")
{
  const auto error = [](int cells, bool slope) {
    const auto space = pcms::CreateSplineSpace(Periodic(cells, TwoPi));
    std::vector<Real> d;
    for (int i = 0; i < cells; ++i) {
      d.push_back(std::sin(TwoPi * i / cells));
    }
    auto s = space->CreateFunction<Real>();
    SetDofs(s, d);
    // interpolation of the node values
    std::vector<std::vector<Real>> nodes;
    for (int i = 0; i < cells; ++i) {
      nodes.push_back({TwoPi * i / cells});
    }
    const auto at_nodes = EvaluateAt(*space, s, nodes);
    for (int i = 0; i < cells; ++i) {
      CHECK(at_nodes[i][0] == Catch::Approx(d[i]).margin(1e-12));
    }
    Real worst = 0;
    const auto pts = Points(60, -1.0, 8.0);
    const auto v = EvaluateAt(*space, s, pts,
                              slope ? pcms::DifferentialOperator::Gradient
                                    : pcms::DifferentialOperator::Value);
    for (std::size_t q = 0; q < pts.size(); ++q) {
      const Real exact = slope ? std::cos(pts[q][0]) : std::sin(pts[q][0]);
      worst = std::max(worst, std::abs(v[q][0] - exact));
    }
    return worst;
  };
  CHECK(error(16, false) < 5e-4);
  CHECK(error(16, false) / error(32, false) > 12.0);
  CHECK(error(32, true) < 2e-3);
  const auto space = pcms::CreateSplineSpace(Periodic(7, 2.0, 1.0));
  auto one = space->CreateFunction<Real>();
  SetDofs(one, std::vector<Real>(7, 3.5));
  const auto v = EvaluateAt(*space, one, Points(9, 0.0, 5.0));
  const auto d = EvaluateAt(*space, one, Points(9, 0.0, 5.0),
                            pcms::DifferentialOperator::Gradient);
  for (std::size_t q = 0; q < v.size(); ++q) {
    CHECK(v[q][0] == Catch::Approx(3.5).margin(1e-12));
    CHECK(d[q][0] == Catch::Approx(0.0).margin(1e-11));
  }
  CHECK_THROWS(pcms::CreateSplineSpace(Periodic(2)));
}

TEST_CASE("Periodic cubic spline: matches the spline of the dense "
          "periodic moment solve",
          "[periodic][spline]")
{
  for (const int n : {8, 64}) {
    CAPTURE(n);
    const Real h = TwoPi / n;
    const auto f = pcms::test::Pseudo(static_cast<std::size_t>(n), 3);
    // M = S f with the dense S
    std::vector<Real> m(n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        Real sum = 0;
        for (int k = 0; k < n; ++k) {
          const Real phi = TwoPi * k / n;
          sum += (2 * std::cos(phi) - 2) / h /
                 (h / 6 * (4 + 2 * std::cos(phi))) * std::cos(phi * (i - j));
        }
        m[i] += sum / n * f[j];
      }
    }
    const auto space = pcms::CreateSplineSpace(Periodic(n));
    auto s = space->CreateFunction<Real>();
    SetDofs(s, f);
    const auto pts = Points(3 * n, 0.0, TwoPi);
    const auto v = EvaluateAt(*space, s, pts);
    for (std::size_t q = 0; q < pts.size(); ++q) {
      const int c = std::min(static_cast<int>(pts[q][0] / h), n - 1);
      const int c1 = (c + 1) % n;
      const Real t = pts[q][0] / h - c;
      const Real u = 1 - t;
      const Real expected =
        u * f[c] + t * f[c1] +
        h * h / 6 * ((u * u * u - u) * m[c] + (t * t * t - t) * m[c1]);
      CHECK(v[q][0] == Catch::Approx(expected).margin(1e-11));
    }
  }
}

TEST_CASE("Periodic bicubic spline: converges at fourth order with either "
          "axis periodic",
          "[periodic][spline]")
{
  const auto chi_axis = GENERATE(0, 1);
  CAPTURE(chi_axis);
  std::vector<std::vector<Real>> pts;
  for (int i = 0; i < 23; ++i) {
    for (int j = 0; j < 29; ++j) {
      pts.push_back(FluxPoint(i / 22.0, TwoPi * j / 28.0, chi_axis));
    }
  }
  std::vector<std::array<Real, 2>> e;
  for (const int n : {10, 20, 40}) {
    const auto [space, f] = FluxSpline(n, chi_axis);
    const auto v = EvaluateAt(*space, f, pts);
    const auto g =
      EvaluateAt(*space, f, pts, pcms::DifferentialOperator::Gradient);
    std::array<Real, 2> worst = {0, 0};
    for (std::size_t q = 0; q < pts.size(); ++q) {
      const auto exact = Flux(pts[q][1 - chi_axis], pts[q][chi_axis]);
      worst[0] = std::max(worst[0], std::abs(v[q][0] - exact[0]));
      worst[1] = std::max({worst[1], std::abs(g[q][1 - chi_axis] - exact[1]),
                           std::abs(g[q][chi_axis] - exact[2])});
    }
    e.push_back(worst);
  }
  CHECK(e[2][0] < 5e-5);
  for (std::size_t i = 1; i < e.size(); ++i) {
    CAPTURE(i, e[i - 1][0], e[i][0], e[i - 1][1], e[i][1]);
    CHECK(std::log2(e[i - 1][0] / e[i][0]) > 3.5);
    CHECK(std::log2(e[i - 1][1] / e[i][1]) > 2.5);
  }
}

TEST_CASE("Periodic bicubic spline: continuous across the seam, and a point "
          "outside the period evaluates at its image",
          "[periodic][spline]")
{
  const auto chi_axis = GENERATE(0, 1);
  CAPTURE(chi_axis);
  const auto [space, f] = FluxSpline(12, chi_axis);
  const Real eps = 1e-9;
  const auto at = [chi_axis](Real chi) {
    std::vector<std::vector<Real>> pts;
    for (const Real x : {0.0, 0.37, 1.0}) {
      pts.push_back(FluxPoint(x, chi, chi_axis));
    }
    return pts;
  };
  for (const auto op : {pcms::DifferentialOperator::Value,
                        pcms::DifferentialOperator::Gradient}) {
    const auto seam = EvaluateAt(*space, f, at(0.0), op);
    const auto below = EvaluateAt(*space, f, at(TwoPi - eps), op);
    const auto above = EvaluateAt(*space, f, at(eps), op);
    const auto outside = EvaluateAt(*space, f, at(-0.3), op);
    const auto inside = EvaluateAt(*space, f, at(TwoPi - 0.3), op);
    for (std::size_t p = 0; p < seam.size(); ++p) {
      for (std::size_t c = 0; c < seam[p].size(); ++c) {
        CAPTURE(p, c);
        CHECK(below[p][c] == Catch::Approx(seam[p][c]).margin(1e-7));
        CHECK(above[p][c] == Catch::Approx(seam[p][c]).margin(1e-7));
        CHECK(outside[p][c] == Catch::Approx(inside[p][c]).margin(1e-12));
      }
    }
  }
}

TEST_CASE("Spline spaces reject twist-shift periodic axes",
          "[periodic][spline]")
{
  pcms::UniformGrid<1> radial;
  radial.bot_left = {0.0};
  radial.edge_length = {1.0};
  radial.divisions = {4};
  const auto shift_space = pcms::CreateLagrangeSpace(
    pcms::uniform_grid::CreateGeometry(radial, pcms::csys::CreateCartesian(1)),
    1, 1);
  const auto shift = std::make_shared<const pcms::Function<Real>>(
    shift_space->CreateFunction<Real>());
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0, 0};
  grid.edge_length = {1, TwoPi};
  grid.divisions = {8, 8};
  const auto geometry = pcms::uniform_grid::CreateGeometry(
    grid, pcms::csys::CreateParameterSystem(
            "index", 2,
            {{1, 0.0, TwoPi, pcms::CreateTwistShiftTransform(0, shift, {0})}}));
  CHECK_THROWS_AS(pcms::CreateSplineSpace(geometry), pcms::pcms_error);
}

TEST_CASE("Periodic spaces: each of two components evaluates as a scalar "
          "field",
          "[periodic][components]")
{
  const auto geometry = Periodic(8, 2.0, 0.5);
  const std::vector<std::pair<std::shared_ptr<const pcms::FunctionSpace>,
                              std::shared_ptr<const pcms::FunctionSpace>>>
    spaces = {
      {pcms::CreateSplineSpace(geometry), pcms::CreateSplineSpace(geometry, 2)},
      {pcms::CreatePeriodicHermiteSpace(geometry),
       pcms::CreatePeriodicHermiteSpace(geometry, 2)},
      {pcms::CreateFourierSpace(geometry, {0, 1, 2}),
       pcms::CreateFourierSpace(geometry, {0, 1, 2}, 1, 2)}};
  const auto pts = Points(7, -1.0, 4.0);
  const auto device = pcms::test::DevicePoints(pts);
  const auto bound =
    geometry->Bind(pcms::CoordinateView<pcms::DeviceMemorySpace>(
      geometry->GetCoordinateSystem(), pcms::MakeConstRank2View(device)));
  for (const auto& [one, two] : spaces) {
    REQUIRE(two->GetLayout()->GetNumComponents() == 2);
    const auto n =
      static_cast<std::size_t>(one->GetLayout()->GetNumDofHolder());
    const std::vector<std::vector<Real>> dofs = {pcms::test::Pseudo(n, 1),
                                                 pcms::test::Pseudo(n, 2)};
    Kokkos::View<Real**, pcms::HostMemorySpace> values("values", n, 2);
    for (std::size_t i = 0; i < n; ++i) {
      values(i, 0) = dofs[0][i];
      values(i, 1) = dofs[1][i];
    }
    auto both = two->CreateFunction<Real>(
      "both", pcms::ValueBasis{.kind = pcms::ValueKind::Array});
    both.SetDOFHolderDataHost(
      pcms::ValueView<const Real, pcms::HostMemorySpace>(
        both.GetData().GetValueBasis(), pcms::MakeConstRank2View(values)));
    const auto evaluator = two->CreatePointEvaluator<Real>(bound);
    Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", pts.size(), 2);
    evaluator->Evaluate(
      both, pcms::ValueView<Real, pcms::DeviceMemorySpace>(
              evaluator->OutputBasis(both.GetData().GetValueBasis()),
              pcms::MakeRank2View(out)));
    const auto host =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, out);
    for (int c = 0; c < 2; ++c) {
      auto scalar = one->CreateFunction<Real>();
      SetDofs(scalar, dofs[c]);
      const auto expected = EvaluateAt(*one, scalar, pts);
      for (std::size_t q = 0; q < pts.size(); ++q) {
        CHECK(host(q, c) == Catch::Approx(expected[q][0]).margin(1e-12));
      }
    }
  }
}

TEST_CASE("Periodic spaces: <E x, y> = <x, E^T y> and the matrix is E",
          "[periodic][adjoint]")
{
  const auto geometry = Periodic(9, 3.0, -1.0);
  const auto points = Points(31, -5.0, 6.0);
  pcms::test::CheckAdjointAndMatrix(
    pcms::CreateFourierSpace(geometry, {0, 1, 2, 5}, 2), points);
  pcms::test::CheckAdjointAndMatrix(pcms::CreateLagrangeSpace(geometry, 1, 1),
                                    points);
  pcms::test::CheckAdjointAndMatrix(pcms::CreatePeriodicHermiteSpace(geometry),
                                    points);
}

TEST_CASE("Periodic spaces: natural quadrature integrates products of two "
          "functions exactly",
          "[periodic][quadrature]")
{
  const auto geometry = Periodic(6, 2.0, 0.5);
  for (const auto& space : {pcms::CreateFourierSpace(geometry, {0, 1, 2, 3}),
                            pcms::CreateLagrangeSpace(geometry, 1, 1)}) {
    const auto quadrature = space->GetQuadrature();
    const auto weights = Kokkos::create_mirror_view_and_copy(
      pcms::HostMemorySpace{}, quadrature.weights);
    Real total = 0;
    for (std::size_t p = 0; p < weights.extent(0); ++p) {
      total += weights(p);
    }
    CHECK(total == Catch::Approx(2.0).margin(1e-13));
  }
  // the mass matrix of Fourier modes is diagonal: 2 pi/period... and CG takes
  // two iterations (two distinct eigenvalues)
  const auto fourier = pcms::CreateFourierSpace(Periodic(4), {0, 1, 2, 3, 4});
  const pcms::transfer::detail::BilinearForm mass(fourier);
  pcms::transfer::detail::LinearSolver solver(mass, pcms::LinearSolveOptions{});
  auto b = fourier->CreateFunction<Real>();
  SetDofs(b, {1, 2, -3, 4, 5, -6, 7, 8, -9});
  auto x = fourier->CreateFunction<Real>();
  CHECK(solver.Solve(b, x) <= 2);
  const auto sol = pcms::test::GetDofs(x);
  CHECK(sol[0] == Catch::Approx(1 / TwoPi).epsilon(1e-12));
  CHECK(sol[1] == Catch::Approx(2 / (TwoPi / 2)).epsilon(1e-12));
}

TEST_CASE("Periodic spaces: Fourier projection reproduces the analytic "
          "coefficients; Lagrange and Hermite projections are accurate",
          "[periodic][projection]")
{
  const auto geometry = Periodic(6);
  const auto source_space = pcms::CreateFourierSpace(geometry, {0, 1, 2});
  auto source = source_space->CreateFunction<Real>();
  const std::vector<Real> c = {0.5, 2.0, -1.0, 0.25, 3.0};
  SetDofs(source, c);
  for (const int harmonics : {1, 2, 4}) {
    CAPTURE(harmonics);
    std::vector<int> modes;
    for (int m = 0; m <= harmonics; ++m) {
      modes.push_back(m);
    }
    const auto target_space = pcms::CreateFourierSpace(geometry, modes);
    pcms::ProjectionOptions options;
    options.quadrature =
      pcms::CreateFourierSpace(geometry, {0, 1, 2, 3, 4, 5, 6})
        ->GetQuadrature();
    auto target = target_space->CreateFunction<Real>();
    pcms::CreateProjection(source_space, target_space, options)
      ->Apply(source, target);
    const auto values = pcms::test::GetDofs(target);
    for (std::size_t m = 0; m < values.size(); ++m) {
      CHECK(values[m] ==
            Catch::Approx(m < c.size() ? c[m] : 0.0).margin(1e-12));
    }
  }
  // sin(x) + 0.3 cos(2 x) onto the Lagrange space of 256 cells and the
  // Hermite space of 16 cells
  const auto f = [](Real x) { return std::sin(x) + 0.3 * std::cos(2 * x); };
  const auto fine_space = pcms::CreateFourierSpace(geometry, {1, 2});
  auto smooth = fine_space->CreateFunction<Real>();
  SetDofs(smooth, {0.0, 1.0, 0.3, 0.0});
  for (const auto& space : {pcms::CreateLagrangeSpace(Periodic(256), 1, 1),
                            pcms::CreatePeriodicHermiteSpace(Periodic(16))}) {
    pcms::ProjectionOptions options;
    options.quadrature = pcms::CreateGaussQuadrature(space->GetGeometry(), 10);
    auto target = space->CreateFunction<Real>();
    pcms::CreateProjection(fine_space, space, options)->Apply(smooth, target);
    const auto pts = Points(40, 0.0, TwoPi);
    const auto v = EvaluateAt(*space, target, pts);
    for (std::size_t q = 0; q < pts.size(); ++q) {
      CHECK(v[q][0] == Catch::Approx(f(pts[q][0])).margin(2e-3));
    }
  }
}

TEST_CASE("Periodic spaces: a projection onto Fourier modes with an inexact "
          "quadrature and a control variate is exact for a function of the "
          "space",
          "[periodic][projection][control_variate]")
{
  const auto geometry = Periodic(6);
  const auto fourier = pcms::CreateFourierSpace(geometry, {0, 1, 2});
  const auto dofs = pcms::test::Pseudo(5, 4);
  auto f = fourier->CreateFunction<Real>();
  SetDofs(f, dofs);
  pcms::ProjectionOptions options;
  options.quadrature = pcms::CreateGaussQuadrature(geometry, 1);
  options.control_variate = true;
  auto g = fourier->CreateFunction<Real>();
  pcms::CreateProjection(fourier, fourier, options)->Apply(f, g);
  const auto out = pcms::test::GetDofs(g);
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    CHECK(out[i] == Catch::Approx(dofs[i]).margin(1e-10));
  }
  // a target without nodal DOFs or a natural quadrature
  const auto bare = std::make_shared<const pcms::FunctionSpace>(
    fourier->GetLayout(), fourier, geometry);
  CHECK_THROWS(pcms::CreateProjection(fourier, bare, options));
}

TEST_CASE("Periodic spaces: a lumped mass needs a non-negative basis",
          "[periodic][projection]")
{
  pcms::ProjectionOptions options;
  options.mass_matrix_type = pcms::MassMatrixType::Lumped;
  const auto fourier = pcms::CreateFourierSpace(Periodic(8), {0, 1});
  CHECK_THROWS_WITH(pcms::CreateProjection(fourier, fourier, options),
                    Catch::Matchers::ContainsSubstring("non-negative basis"));
  const auto hermite = pcms::CreatePeriodicHermiteSpace(Periodic(8));
  CHECK_THROWS_WITH(pcms::CreateProjection(hermite, hermite, options),
                    Catch::Matchers::ContainsSubstring("non-negative basis"));
}

TEST_CASE("Periodic Lagrange: the matrix-free projection onto a refined "
          "space conserves the integral",
          "[periodic][projection]")
{
  // the target cells halve the source cells, so the natural quadrature of
  // the target integrates (source) x (target basis) exactly and the
  // projection of a field whose space holds the constants keeps its integral
  const auto source_space =
    pcms::CreateLagrangeSpace(Periodic(8, 3.0, 1.0), 1, 1);
  const auto target_space =
    pcms::CreateLagrangeSpace(Periodic(16, 3.0, 1.0), 1, 1);
  auto source = source_space->CreateFunction<Real>();
  SetDofs(source, {1, 4, -2, 0.5, 3, 7, -1, 2});
  const Real expected = pcms::L2Integral(source_space).Apply(source);
  CHECK(expected == Catch::Approx(3.0 / 8 * 14.5).epsilon(1e-14));
  const pcms::L2Integral integral(target_space);
  for (const auto mass :
       {pcms::MassMatrixType::Consistent, pcms::MassMatrixType::Lumped}) {
    CAPTURE(static_cast<int>(mass));
    pcms::ProjectionOptions options;
    options.mass_matrix_type = mass;
    options.solve.relative_tolerance = 1e-14;
    auto target = target_space->CreateFunction<Real>();
    pcms::CreateProjection(source_space, target_space, options)
      ->Apply(source, target);
    CHECK(integral.Apply(target) == Catch::Approx(expected).epsilon(1e-13));
  }
}

TEST_CASE("Twist-shift periodic axis: points just across the seam evaluate "
          "as the shifted interior, in either periodic axis order",
          "[periodic][twist]")
{
  const bool translation_first = GENERATE(false, true);
  CAPTURE(translation_first);
  // S(x) = 0.3 + 0.5 x on a radial grid in another system
  pcms::UniformGrid<1> radial;
  radial.bot_left = {0.0};
  radial.edge_length = {1.0};
  radial.divisions = {4};
  const auto s_space = pcms::CreateLagrangeSpace(
    pcms::uniform_grid::CreateGeometry(radial, pcms::csys::CreateCartesian(1)),
    1, 1);
  auto shift = s_space->CreateFunction<Real>();
  SetDofs(shift, {0.3, 0.425, 0.55, 0.675, 0.8});
  const auto s = [](Real x) { return 0.3 + 0.5 * x; };
  // index coordinates (x, y, z): y twist-shifted by S(x), z periodic
  const pcms::PeriodicAxis twist{
    1, 0.0, 1.0,
    pcms::CreateTwistShiftTransform(
      2, std::make_shared<const pcms::Function<Real>>(std::move(shift)), {0})};
  const pcms::PeriodicAxis translation{2, 0.0, TwoPi};
  const auto index = pcms::csys::CreateParameterSystem(
    "index", 3,
    translation_first ? std::vector{translation, twist}
                      : std::vector{twist, translation});
  pcms::UniformGrid<3> grid;
  grid.bot_left = {0.0, 0.0, 0.0};
  grid.edge_length = {1.0, 1.0, TwoPi};
  grid.divisions = {4, 4, 8};
  const auto space = pcms::CreateLagrangeSpace(
    pcms::uniform_grid::CreateGeometry(grid, index), 1, 1);
  // y keeps its last node, z drops it
  REQUIRE(space->GetLayout()->GetNumDofHolder() == 5 * 5 * 8);
  std::vector<Real> dofs;
  for (int i = 0; i < 200; ++i) {
    dofs.push_back(std::sin(1.7 * i) + 0.1 * i);
  }
  auto u = space->CreateFunction<Real>();
  SetDofs(u, dofs);
  const Real eps = 0.01;
  for (const Real x : {0.1, 0.55, 0.9}) {
    for (const Real z : {0.2, 3.0, TwoPi - 0.1}) {
      CAPTURE(x, z);
      const auto across =
        EvaluateAt(*space, u, {{x, 1 + eps, z}, {x, -eps, z}, {x, 2 + eps, z}});
      const auto interior = EvaluateAt(*space, u,
                                       {{x, eps, z + s(x)},
                                        {x, 1 - eps, z - s(x) + TwoPi},
                                        {x, eps, z + 2 * s(x)}});
      for (int p = 0; p < 3; ++p) {
        CHECK(across[p][0] == Catch::Approx(interior[p][0]).margin(1e-12));
      }
    }
  }
  // outside the domain of S, the policy of the bind applies
  const auto geometry = space->GetGeometry();
  const auto outside = pcms::test::DevicePoints({{1.5, 0.5, 1.0}});
  const auto bound = geometry->Bind(
    pcms::CoordinateView<pcms::DeviceMemorySpace>(
      geometry->GetCoordinateSystem(), pcms::MakeConstRank2View(outside)),
    pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL});
  CHECK(bound->MissingPoints().extent(0) == 1);
  CHECK_THROWS(pcms::CreateTwistShiftTransform(2, nullptr, {0}));
}
