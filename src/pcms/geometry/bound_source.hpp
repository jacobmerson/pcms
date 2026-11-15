#ifndef PCMS_GEOMETRY_BOUND_SOURCE_HPP
#define PCMS_GEOMETRY_BOUND_SOURCE_HPP

#include "pcms/field/coordinate_view.hpp"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/geometry/parametric_points.hpp"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

class Geometry;
class FieldLayout;
class BoundCoordinateMap;

/// @brief What Geometry::Bind hands to BindImpl.
struct BindRequest
{
  /// points to locate, in the geometry's coordinate system: the query points,
  /// or their images under `map`; may be released after BindImpl returns
  CoordinateView<DeviceMemorySpace> coords;
  /// layout whose DOF-holder coordinates `query_coords` are, or null (used by
  /// localization path selection until S5 removes it)
  std::shared_ptr<const FieldLayout> query_layout;
  /// out-of-bounds policy of the evaluators created from the bound source
  OutOfBoundsPolicy policy;
  /// the coordinate map bound to `query_coords`, or null if the query points
  /// are in the geometry's coordinate system
  std::shared_ptr<const BoundCoordinateMap> map;
  /// the query points as given to Bind; `coords` itself without a map
  CoordinateView<DeviceMemorySpace> query_coords;
};

/// @brief A set of query points localized on a geometry.
///
/// Created by Geometry::Bind. Valid for every evaluator source whose
/// GetGeometry() is this source's geometry, and stays valid after later binds
/// on the same geometry. Immutable after construction. It is shared by every
/// evaluator created from it; evaluators that share a bound source must not be
/// evaluated concurrently.
class BoundSource
{
public:
  /// @return the geometry the points were localized on
  [[nodiscard]] const std::shared_ptr<const Geometry>& GetGeometry()
    const noexcept
  {
    return geometry_;
  }

  /// @return number of bound query points
  [[nodiscard]] LO NumPoints() const noexcept { return num_points_; }

  /// @return out-of-bounds policy of the evaluators created from this source
  [[nodiscard]] const OutOfBoundsPolicy& Policy() const noexcept
  {
    return policy_;
  }

  /// @return layout whose DOF holders are the bound points, or null if the
  ///         source was bound to coordinates
  [[nodiscard]] const FieldLayout* QueryLayout() const noexcept
  {
    return query_layout_.get();
  }

  /// @return coordinate system of PointValues(): the query system, which is
  ///         the geometry's system unless the bind used a coordinate map
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept
  {
    return system_;
  }

  /// @return the coordinate map bound to the query points, or null if the
  ///         bind had no map. Evaluators created from this source evaluate at
  ///         the mapped points and re-express their values through it.
  [[nodiscard]] const std::shared_ptr<const BoundCoordinateMap>& GetBoundMap()
    const noexcept
  {
    return map_;
  }

  /// @return whether the bind had physical query points; false for a
  ///         parametric bind, whose PointValues() is empty
  [[nodiscard]] bool HasPointValues() const noexcept
  {
    return points_.extent(0) > 0;
  }

  /// @return the coordinates of the bound query points in
  ///         GetCoordinateSystem() (a copy owned by this object), one row per
  ///         point; empty for a parametric bind
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace> PointValues()
    const noexcept
  {
    return points_;
  }

  /// @return the bound points in the geometry's coordinate system, where they
  ///         were located: PointValues() mapped by GetBoundMap(), or
  ///         PointValues() itself if the bind had no map; empty for a
  ///         parametric bind
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace>
  LocatedPointValues() const noexcept
  {
    return located_;
  }

  /// @return indices of the query points the source evaluates, ascending
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> FoundPoints()
    const noexcept
  {
    return found_;
  }

  /// @return indices of the query points the source does not evaluate,
  ///         ascending; their rows are filled in OutOfBoundsMode::FILL
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> MissingPoints()
    const noexcept
  {
    return missing_;
  }

  virtual ~BoundSource() noexcept = default;

  BoundSource(const BoundSource&) = delete;
  BoundSource& operator=(const BoundSource&) = delete;

protected:
  /// @brief Copies the request's query points (and, with a map, the mapped
  ///        points); every point counts as found until SetFound is called.
  /// @param geometry geometry the points are localized on
  /// @param request bound query points, map and out-of-bounds policy
  BoundSource(std::shared_ptr<const Geometry> geometry,
              const BindRequest& request);

  /// @brief Points given in parametric coordinates: no physical coordinates,
  ///        every point found.
  /// @param geometry geometry the points are localized on
  /// @param points parametric points
  /// @param policy out-of-bounds policy of the evaluators created from this
  ///        source
  BoundSource(std::shared_ptr<const Geometry> geometry,
              const ParametricPoints& points, OutOfBoundsPolicy policy);

  /// @brief Records which query points the source evaluates.
  /// @param found per query point, whether the source evaluates it
  /// @throws pcms_error in OutOfBoundsMode::ERROR if a point is not found
  void SetFound(Kokkos::View<const bool*, DeviceMemorySpace> found);

  /// @brief Takes the found and missing points of a bound source over the
  ///        same number of points.
  /// @param same_points bound source whose points correspond to these
  void SetFound(const BoundSource& same_points);

private:
  std::shared_ptr<const Geometry> geometry_;
  OutOfBoundsPolicy policy_;
  std::shared_ptr<const FieldLayout> query_layout_;
  std::shared_ptr<const BoundCoordinateMap> map_;
  std::shared_ptr<const CoordinateSystem> system_;
  LO num_points_;
  Kokkos::View<Real**, DeviceMemorySpace> points_;
  Kokkos::View<Real**, DeviceMemorySpace> located_;
  Kokkos::View<LO*, DeviceMemorySpace> found_;
  Kokkos::View<LO*, DeviceMemorySpace> missing_;
};

} // namespace pcms

#endif // PCMS_GEOMETRY_BOUND_SOURCE_HPP
