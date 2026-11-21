#ifndef PCMS_GEOMETRY_QUADRATURE_HPP
#define PCMS_GEOMETRY_QUADRATURE_HPP

#include "pcms/geometry/bound_points.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <memory>

namespace pcms
{

/// @brief The rule Geometry::CreateQuadrature generates: deterministic of
///        `order`, or Monte Carlo when `samples_per_cell` is positive.
struct QuadratureOptions
{
  /// degree the deterministic rule integrates exactly on each cell
  int order = 2;
  /// uniform random samples per cell, each weighing cell measure / samples
  int samples_per_cell = 0;
  /// seed of the Monte Carlo rule
  std::uint64_t seed = 8675309;
};

/// @brief Points with weights such that the integral of f over the cells is
///        sum_p weights(p) f(x_p).
struct Quadrature
{
  /// the points, bound on the geometry that generated them
  std::shared_ptr<const BoundPoints> points;
  /// one weight per point, in the physical measure
  Kokkos::View<const Real*, DeviceMemorySpace> weights;
  /// cells of a second geometry the points lie in; empty if there is none
  OwnerHints other;
};

} // namespace pcms

#endif // PCMS_GEOMETRY_QUADRATURE_HPP
