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

/// @brief Which rule Geometry::CreateQuadrature generates.
///
/// A deterministic rule of `order` by default; a Monte Carlo rule when
/// `samples_per_cell` is positive. (An alternative shape is a rule enum
/// with the options of each rule next to it; two fields keep the struct a
/// plain aggregate for the two rules there are.)
struct QuadratureOptions
{
  /// polynomial degree the deterministic rule integrates exactly on each
  /// cell; ignored for the Monte Carlo rule
  int order = 2;
  /// when positive, the rule has this many uniform random samples per cell,
  /// each weighing cell measure / samples_per_cell
  int samples_per_cell = 0;
  /// seed of the Monte Carlo rule
  std::uint64_t seed = 8675309;
};

/// @brief Points on a geometry with the weights that make a sum over them an
///        integral over the geometry's cells: the integral of f is
///        sum_p weights(p) f(x_p).
///
/// The points are generated in the geometry's cells, so they are bound
/// without a search; `weights` carry the cell measure (|det J| of the
/// geometry's differential). `other` optionally names the cells of a second
/// geometry the points also lie in (the quadrature of a pair of geometries,
/// e.g. the intersection of two meshes), to bind the second geometry to the
/// points without a search either.
struct Quadrature
{
  /// the points, bound on the geometry that generated them
  std::shared_ptr<const BoundPoints> points;
  /// one weight per point
  Kokkos::View<const Real*, DeviceMemorySpace> weights;
  /// cells of a second geometry the points lie in; empty if there is none
  OwnerHints other;
};

} // namespace pcms

#endif // PCMS_GEOMETRY_QUADRATURE_HPP
