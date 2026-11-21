#ifndef PCMS_UTILITY_COORDINATE_SYSTEMS_EUCLIDEAN_HPP
#define PCMS_UTILITY_COORDINATE_SYSTEMS_EUCLIDEAN_HPP

#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/types.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace pcms::csys
{

/// One axis of a Euclidean coordinate system.
struct Axis
{
  /// label, not used by PCMS
  std::string name;
  /// period (> 0) if the axis is periodic
  std::optional<Real> period;
  /// weight (> 0) of the axis in point-cloud distances
  Real scale = 1.0;
  bool operator==(const Axis&) const = default;
};

/// @brief Non-spatial coordinates (e.g. phase space) with identity metric.
///        Point-cloud geometries measure distances in them using the periods
///        and scales of the axes; the search radius is in scaled units.
/// @param axes the axes
/// @return the coordinate system; it is not Cartesian
/// @throws pcms_error unless there are 1..6 axes with positive finite periods
///         and scales
[[nodiscard]] std::shared_ptr<const CoordinateSystem> CreateEuclidean(
  std::vector<Axis> axes);

/// @brief As above with `dim` axes that are neither periodic nor scaled.
/// @throws pcms_error unless `dim` is in 1..6
[[nodiscard]] std::shared_ptr<const CoordinateSystem> CreateEuclidean(int dim);

/// @return the axes of a system made by CreateEuclidean, empty for any other
[[nodiscard]] std::vector<Axis> GetAxes(const CoordinateSystem& system);

} // namespace pcms::csys

#endif // PCMS_UTILITY_COORDINATE_SYSTEMS_EUCLIDEAN_HPP
