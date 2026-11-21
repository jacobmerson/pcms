#ifndef PCMS_LOCALIZATION_MLS_SUPPORT_HELPERS_HPP
#define PCMS_LOCALIZATION_MLS_SUPPORT_HELPERS_HPP

#include "pcms/geometry/supports.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"

#include <Kokkos_Core.hpp>

namespace pcms::detail
{

/// Periods and scales of the axes of a point cloud's coordinates; identity by
/// default.
struct AxisMetric
{
  static constexpr int MaxDim = 6;
  Real period[MaxDim] = {}; // 0: not periodic
  Real scale[MaxDim] = {1, 1, 1, 1, 1, 1};

  /// @return scale * (to - from) along axis `d`, with `to` at the periodic
  ///         image nearest `from`
  KOKKOS_INLINE_FUNCTION Real Offset(Real from, Real to, int d) const
  {
    Real x = to - from;
    if (period[d] > 0) {
      x -= period[d] * Kokkos::round(x / period[d]);
    }
    return scale[d] * x;
  }
};

/// @return the metric of the axes of a system made by csys::CreateEuclidean,
///         the identity for any other
AxisMetric MakeAxisMetric(const CoordinateSystem& system);

// Returns true when both the minimum and maximum support counts fall within
// the requested range.
inline bool within_number_of_support_range(unsigned min_supports_found,
                                           unsigned max_supports_found,
                                           unsigned min_req_supports,
                                           unsigned max_allowed_supports)
{
  return (min_supports_found >= min_req_supports) &&
         (max_supports_found <= max_allowed_supports);
}

// Compute the element-wise min and max of a support-count array.
void minmax(Kokkos::View<const LO*, DeviceMemorySpace> num_supports,
            unsigned& min_supports_found, unsigned& max_supports_found);

// Scale per-target squared radii to drive support counts toward [min, max].
void adapt_radii(unsigned min_req_supports, unsigned max_allowed_supports,
                 LO n_targets, Kokkos::View<Real*, DeviceMemorySpace> radii2_l,
                 Kokkos::View<LO*, DeviceMemorySpace> num_supports);

// Build SupportResults for a point-cloud source/target pair using a
// distance-based N² search with optional adaptive radius adjustment.
// source_coords and target_coords are flat arrays in [x0,y0,..., x1,y1,...]
// order.
SupportResults BuildPointCloudSupports(
  Kokkos::View<const Real*, DeviceMemorySpace> source_coords,
  Kokkos::View<const Real*, DeviceMemorySpace> target_coords, int dim,
  double radius, unsigned min_req_supports, bool adapt_radius,
  unsigned max_iterations = 100, const AxisMetric& metric = {});

} // namespace pcms::detail
#endif // PCMS_LOCALIZATION_MLS_SUPPORT_HELPERS_HPP
