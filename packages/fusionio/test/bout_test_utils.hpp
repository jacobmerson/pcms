#ifndef FUSIONIO_TEST_BOUT_TEST_UTILS_HPP
#define FUSIONIO_TEST_BOUT_TEST_UTILS_HPP

#include "fusionio/bout.hpp"
#include "test_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numbers>
#include <vector>

namespace fusionio::test
{

/// `function` at points in index coordinates, on the host.
inline Kokkos::View<Real**, HostMemorySpace> AtIndex(
  const pcms::Function<Real>& function, const std::vector<Real>& points)
{
  const auto& geometry = *function.GetSpace().GetGeometry();
  const auto system = geometry.GetParameterSystem()
                        ? geometry.GetParameterSystem()
                        : geometry.GetCoordinateSystem();
  const auto coords = CreateDeviceCoordinateView(points, system, 3);
  return EvaluateWith(*function.GetSpace().CreatePointEvaluator<Real>(
                        geometry.Bind(coords.coordinate_view)),
                      function);
}

/// Ensure that points just past the y seam have the position of the points
/// just before it (phi up to the toroidal period).
inline void CheckSeamPosition(const BoutSpaces& spaces, Real nyi, Real x,
                              Real z, Real phi_period, Real tol)
{
  const Real eps = 1e-6;
  const auto pos =
    AtIndex(*spaces.geometry->GetPosition(),
            {x, nyi - eps, z, x, nyi + eps, z, x, -eps, z, x, eps, z});
  for (int pair = 0; pair < 2; ++pair) {
    const int a = 2 * pair;
    CAPTURE(x, z, pair);
    CHECK_THAT(pos(a + 1, 0), Catch::Matchers::WithinAbs(pos(a, 0), tol));
    CHECK_THAT(pos(a + 1, 2), Catch::Matchers::WithinAbs(pos(a, 2), tol));
    const Real dphi = std::remainder(pos(a + 1, 1) - pos(a, 1), phi_period);
    CHECK_THAT(dphi, Catch::Matchers::WithinAbs(0.0, tol));
  }
}

} // namespace fusionio::test

#endif // FUSIONIO_TEST_BOUT_TEST_UTILS_HPP
