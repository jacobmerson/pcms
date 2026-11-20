#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fusionio/geqdsk.hpp>
#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::Quantity;
using fusionio::Real;
using fusionio::test::EvaluateAt;

namespace
{

// Fusion-IO's example geqdsk file (EFIT, DIII-D #158115, 129 x 129).
const std::string GeqdskFile =
  std::string(FUSIONIO_TEST_DATA_DIR) + "/geqdsk/g158115.04701";

// The grid of the file: R from 0.84 over 1.7 and Z from -1.6 over 3.2 in 128
// cells each.
constexpr Real RLeft = 0.84, RDim = 1.7, ZBottom = -1.6, ZDim = 3.2;
constexpr int Cells = 128;

// n (R, phi, Z) points in the grid, `margin` cells from its edge (the
// bicubic interpolation of Fusion-IO 1.x does not reach the outermost two).
std::vector<Real> RandomPoints(int n, int margin)
{
  std::mt19937 generator(7);
  std::uniform_real_distribution<Real> unit(0.0, 1.0);
  const Real fraction = Real(margin) / Cells;
  std::vector<Real> points;
  for (int i = 0; i < n; ++i) {
    points.push_back(RLeft +
                     RDim * (fraction + (1 - 2 * fraction) * unit(generator)));
    points.push_back(6.0 * unit(generator));
    points.push_back(ZBottom +
                     ZDim * (fraction + (1 - 2 * fraction) * unit(generator)));
  }
  return points;
}

} // namespace

TEST_CASE("geqdsk file: the equilibrium is consistent")
{
  const auto source = fusionio::OpenGeqdsk(GeqdskFile);
  const Real axis_r =
    *source->EvaluateSeries(fusionio::Series::MagneticAxisR, 0);
  const Real axis_z =
    *source->EvaluateSeries(fusionio::Series::MagneticAxisZ, 0);
  REQUIRE_THAT(axis_r, WithinAbs(1.73434, 1e-5));
  const auto norm = source->CreateFunction(Quantity::PoloidalFluxNorm);

  // psi_N is 0 at the magnetic axis, which is a stationary point of psi
  REQUIRE_THAT(EvaluateAt(norm, {axis_r, 0.0, axis_z})(0, 0),
               WithinAbs(0.0, 1e-4));
  const auto grad = fusionio::test::EvaluateGradientAt(
    source->CreateFunction(Quantity::PoloidalFlux), {axis_r, 0.0, axis_z});
  REQUIRE_THAT(grad(0, 0), WithinAbs(0.0, 1e-2));
  REQUIRE_THAT(grad(0, 2), WithinAbs(0.0, 1e-2));

  // and 1 on the plasma boundary
  const auto& boundary = source->Boundary();
  REQUIRE(boundary.size() > 10);
  REQUIRE(source->Limiter().size() > 10);
  std::vector<Real> points;
  for (const auto& [r, z] : boundary) {
    points.insert(points.end(), {r, 0.0, z});
  }
  const auto values = EvaluateAt(norm, points);
  for (size_t i = 0; i < boundary.size(); ++i) {
    CAPTURE(i);
    REQUIRE_THAT(values(i, 0), WithinAbs(1.0, 2e-3));
  }

  // The profiles cover [psi_axis, psi_lcfs]; q is positive and rises outward.
  const auto q = source->CreateProfile(fusionio::GeqdskProfile::SafetyFactor);
  const Real psi_axis =
    *source->EvaluateSeries(fusionio::Series::MagneticAxisPsi, 0);
  const Real psi_lcfs = *source->EvaluateSeries(fusionio::Series::LcfsPsi, 0);
  const auto q_v =
    EvaluateAt(q, {psi_axis, psi_axis + 0.5 * (psi_lcfs - psi_axis),
                   psi_axis + 0.999 * (psi_lcfs - psi_axis)});
  REQUIRE(std::abs(q_v(0, 0)) > 0.5);
  REQUIRE(std::abs(q_v(0, 0)) < std::abs(q_v(1, 0)));
  REQUIRE(std::abs(q_v(1, 0)) < std::abs(q_v(2, 0)));
}

#ifdef FUSIONIO_ENABLE_WRAPPED
// The native source against Fusion-IO 1.x on the same file. Both interpolate
// the grid values, so they agree at the grid nodes; between them 1.x uses the
// cubic Lagrange interpolation of the 4 x 4 nearest values (whose derivatives
// jump at the cell edges) and PCMS bicubic splines (not-a-knot, C2), which
// differ by the interpolation error of either. Measured on this file: the
// largest differences relative to the largest value are 3e-4 (flux), 1e-4
// (pressure), 4e-3 (magnetic field) and 2e-2 (flux gradient), at a few points
// of the grid's strongest curvature.
namespace
{

// The largest difference of the components of two evaluations, relative to
// the largest value of the second.
Real MaxRelativeDifference(const Kokkos::View<Real**, pcms::HostMemorySpace>& a,
                           const Kokkos::View<Real**, pcms::HostMemorySpace>& b)
{
  Real difference = 0, scale = 0;
  for (size_t i = 0; i < a.extent(0); ++i) {
    for (size_t c = 0; c < a.extent(1); ++c) {
      difference = std::max(difference, std::abs(a(i, c) - b(i, c)));
      scale = std::max(scale, std::abs(b(i, c)));
    }
  }
  return difference / scale;
}

} // namespace

TEST_CASE("geqdsk file: native and Fusion-IO 1.x agree")
{
  const auto native = fusionio::OpenGeqdsk(GeqdskFile);
  const auto wrapped = fusionio::Open(fusionio::Format::GEQDSK, GeqdskFile);

  SECTION("at the grid nodes")
  {
    std::vector<Real> points;
    for (int i : {5, 40, 64, 77, 120}) {
      for (int j : {4, 33, 64, 100, 123}) {
        points.insert(points.end(), {RLeft + RDim * i / Cells, 0.4,
                                     ZBottom + ZDim * j / Cells});
      }
    }
    for (const Quantity quantity :
         {Quantity::PoloidalFlux, Quantity::PoloidalFluxNorm}) {
      const auto n = EvaluateAt(native->CreateFunction(quantity), points);
      const auto w = EvaluateAt(wrapped->CreateFunction(quantity), points);
      CAPTURE(fusionio::Name(quantity));
      REQUIRE(MaxRelativeDifference(n, w) < 1e-12);
    }
  }

  SECTION("between the nodes")
  {
    const auto points = RandomPoints(2000, 3);
    struct Case
    {
      Quantity quantity;
      Real tolerance;
    };
    for (const auto& [quantity, tolerance] :
         {Case{Quantity::PoloidalFlux, 5e-4},
          Case{Quantity::PoloidalFluxNorm, 5e-4},
          Case{Quantity::TotalPressure, 2e-4},
          Case{Quantity::MagneticField, 1e-2}}) {
      const auto n = EvaluateAt(native->CreateFunction(quantity), points);
      const auto w = EvaluateAt(wrapped->CreateFunction(quantity), points);
      const Real difference = MaxRelativeDifference(n, w);
      std::cout << "native vs 1.x, " << fusionio::Name(quantity)
                << ": max difference / max value = " << difference << "\n";
      CAPTURE(fusionio::Name(quantity));
      REQUIRE(difference < tolerance);
    }
    const auto gradient = [&](const fusionio::Source& source) {
      return fusionio::test::EvaluateGradientAt(
        source.CreateFunction(Quantity::PoloidalFlux), points);
    };
    const Real difference =
      MaxRelativeDifference(gradient(*native), gradient(*wrapped));
    std::cout << "native vs 1.x, gradient of the poloidal flux: max "
                 "difference / max value = "
              << difference << "\n";
    REQUIRE(difference < 5e-2);
  }
}
#endif
