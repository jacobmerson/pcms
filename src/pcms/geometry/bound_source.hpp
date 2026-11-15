#ifndef PCMS_GEOMETRY_BOUND_SOURCE_HPP
#define PCMS_GEOMETRY_BOUND_SOURCE_HPP

#include "pcms/field/coordinate_view.hpp"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

class Geometry;
class BoundCoordinateMap;

/// @brief Entities of a geometry the query points are known to lie in, so a
///        bind or neighbourhood search can start from them instead of
///        searching.
///
/// A hint is trusted: the caller guarantees that each point lies in its
/// entity. The ids are checked to be entities of the geometry, but a stale
/// hint within range is not detected; the point is then evaluated by
/// extrapolation from the given entity (OutOfBoundsMode::ERROR does not
/// throw and FILL does not fill for it).
struct KnownOwners
{
  /// geometry whose entities `entities` are; a hint for another geometry is
  /// ignored
  std::shared_ptr<const Geometry> geometry;
  /// the geometry's top dimension for "point i is in cell entities(i)", 0 for
  /// "point i is at vertex entities(i)"
  int entity_dim = -1;
  /// one entity per query point; empty means no hint
  Kokkos::View<const LO*, DeviceMemorySpace> entities;

  /// @return whether this is a hint for each of `num_points` points on
  ///         `on`
  [[nodiscard]] bool Hints(const Geometry& on, LO num_points) const noexcept
  {
    return geometry.get() == &on && num_points > 0 &&
           static_cast<LO>(entities.extent(0)) == num_points;
  }
};

/// @brief What Geometry::Bind hands to BindImpl.
struct BindRequest
{
  /// points to locate, in the geometry's coordinate system: the query points,
  /// or their images under `map`; may be released after BindImpl returns
  CoordinateView<DeviceMemorySpace> coords;
  /// out-of-bounds policy of the evaluators created from the bound source
  OutOfBoundsPolicy policy;
  /// the owner of each point, from the caller, when they are owners on this
  /// geometry; otherwise empty, and the kind searches
  KnownOwners owners;
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

  /// @return the coordinates of the bound query points in
  ///         GetCoordinateSystem() (a copy owned by this object), one row per
  ///         point
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace> PointValues()
    const noexcept
  {
    return points_;
  }

  /// @return the bound points in the geometry's coordinate system, where they
  ///         were located: PointValues() mapped by GetBoundMap(), or
  ///         PointValues() itself if the bind had no map
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
