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
  /// optional label (not used by PCMS)
  std::string name;
  /// the axis is periodic with this period (> 0): points whose coordinates
  /// differ by a multiple of it are the same point
  std::optional<Real> period;
  /// weight (> 0) of the axis in distances: the point-cloud support search and
  /// MLS work on `scale` * (coordinate difference), so axes of different
  /// magnitude (radians, eV) can share one radius and a well-conditioned fit
  Real scale = 1.0;
};

/// Non-spatial coordinates (e.g. phase space: energy, pitch, ...) with the
/// identity metric, one Axis per coordinate (1..6). It is not Cartesian: the
/// coordinate maps of physical space (Cartesian <-> cylindrical, plane
/// embeddings) refuse it. Point-cloud geometries (point_cloud::CreateGeometry)
/// measure distances in it with the periods and scales of the axes: a support
/// across a period is used at its periodic image nearest the query point, and
/// the search radius is in scaled units (so it must be smaller than half of
/// each scaled period).
/// @throws pcms_error unless there are 1..6 axes, each period (if any) and
///         scale is positive and finite
[[nodiscard]] std::shared_ptr<const CoordinateSystem> CreateEuclidean(
  std::vector<Axis> axes);

/// As above, `dim` axes that are neither periodic nor scaled.
/// @throws pcms_error unless `dim` is in 1..6
[[nodiscard]] std::shared_ptr<const CoordinateSystem> CreateEuclidean(int dim);

/// @return the axes of a system made by CreateEuclidean; empty for any other
///         system (which has no periods or scales)
[[nodiscard]] std::vector<Axis> GetAxes(const CoordinateSystem& system);

} // namespace pcms::csys

#endif // PCMS_UTILITY_COORDINATE_SYSTEMS_EUCLIDEAN_HPP
