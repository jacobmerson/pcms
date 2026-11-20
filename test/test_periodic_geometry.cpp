#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <pcms/geometry/periodic_geometry.hpp>
#include <pcms/utility/arrays.hpp>
#include <pcms/utility/detail/arrays.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include "field_test_utils.hpp"
#include "pcms/geometry/periodic_binding.hpp"
#include <cmath>
#include <numbers>

namespace
{

using pcms::Real;
constexpr Real TwoPi = 2 * std::numbers::pi_v<Real>;

// sum_p w_p f(x_p)
template <typename F>
Real Integrate(const pcms::Quadrature& quadrature, F f)
{
  const auto points = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace{}, quadrature.points->LocatedPointValues());
  const auto weights = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace{}, quadrature.weights);
  Real sum = 0;
  for (std::size_t p = 0; p < points.extent(0); ++p) {
    sum += weights(p) * f(points(p, 0));
  }
  return sum;
}

} // namespace

TEST_CASE("Periodic geometry: points wrap modulo the period at Bind",
          "[periodic]")
{
  const auto geometry =
    pcms::CreatePeriodicGeometry(pcms::csys::CreateCartesian(1), 4, 2.0, 1.0);
  Kokkos::View<Real**, pcms::HostMemorySpace> host("p", 5, 1);
  const Real x[5] = {1.0, 1.6, 0.9, 7.25, -0.4};
  for (int i = 0; i < 5; ++i) {
    host(i, 0) = x[i];
  }
  const auto points = pcms::detail::ToDeviceView(host);
  const auto bound =
    geometry->Bind(pcms::CoordinateView<pcms::DeviceMemorySpace>(
      geometry->GetCoordinateSystem(), pcms::MakeConstRank2View(points)));
  CHECK(bound->MissingPoints().extent(0) == 0);
  const auto& binding =
    dynamic_cast<const pcms::detail::PeriodicBinding&>(*bound);
  const auto phase = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace{}, binding.Phase());
  const auto cell = Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{},
                                                        binding.CellIndices());
  const auto xi = Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{},
                                                      binding.Parametric());
  // [1, 3), h = 0.5
  const int expected_cell[5] = {0, 1, 3, 0, 1};
  const Real expected_xi[5] = {0.0, 0.2, 0.8, 0.5, 0.2};
  const Real expected_phase[5] = {0.0, 0.3, 0.95, 0.125, 0.3};
  for (int i = 0; i < 5; ++i) {
    CHECK(cell(i) == expected_cell[i]);
    CHECK(xi(i) == Catch::Approx(expected_xi[i]).margin(1e-13));
    CHECK(phase(i) == Catch::Approx(expected_phase[i]).margin(1e-13));
  }
}

TEST_CASE("Periodic geometry: Jacobian is the cell size", "[periodic]")
{
  const auto geometry =
    pcms::CreatePeriodicGeometry(pcms::csys::CreateCartesian(1), 5, 2.0);
  const auto quadrature = geometry->CreateQuadrature({.order = 2});
  const auto differential = quadrature.points->Differential();
  REQUIRE(differential.jacobian.extent(0) ==
          static_cast<std::size_t>(quadrature.points->NumPoints()));
  const auto det = pcms::test::ToHostFlat(differential.determinant);
  CHECK(det(0) == Catch::Approx(0.4));
}

TEST_CASE("Periodic geometry: Gauss per cell integrates polynomials of its "
          "order exactly",
          "[periodic][quadrature]")
{
  const auto geometry =
    pcms::CreatePeriodicGeometry(pcms::csys::CreateCartesian(1), 3, 3.0, -1.0);
  for (int order = 0; order <= 9; ++order) {
    CAPTURE(order);
    const auto quadrature = geometry->CreateQuadrature({.order = order});
    // integral of x^order over [-1, 2)
    const Real exact =
      (std::pow(2.0, order + 1) - std::pow(-1.0, order + 1)) / (order + 1);
    CHECK(Integrate(quadrature, [&](Real x) { return std::pow(x, order); }) ==
          Catch::Approx(exact).margin(1e-12));
  }
}

TEST_CASE("Periodic geometry: the trapezoid rule is exact for trigonometric "
          "polynomials",
          "[periodic][quadrature]")
{
  const auto geometry =
    pcms::CreatePeriodicGeometry(pcms::csys::CreateCartesian(1), 4);
  const int n = 9; // exact up to degree 8
  const auto quadrature = pcms::CreateTrapezoidQuadrature(geometry, n);
  for (int j = 0; j <= 4; ++j) {
    for (int k = 0; k <= 4; ++k) {
      CAPTURE(j, k);
      const Real cc = Integrate(
        quadrature, [&](Real x) { return std::cos(j * x) * std::cos(k * x); });
      const Real expected = j != k ? 0.0 : (j == 0 ? TwoPi : TwoPi / 2);
      CHECK(cc == Catch::Approx(expected).margin(1e-12));
    }
  }
  // the rule of degree n - 1 is not exact beyond it
  CHECK(Integrate(quadrature, [](Real x) { return std::cos(9 * x); }) ==
        Catch::Approx(TwoPi).margin(1e-12));
}
