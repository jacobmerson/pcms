#include "pcms/geometry/geometry.hpp"
#include "pcms/field/coordinate_map.hpp"
#include "pcms/localization/adj_search.hpp" // SupportResults
#include "pcms/utility/assert.h"
#include <string>

namespace pcms
{

std::shared_ptr<const BoundSource> Geometry::Bind(
  CoordinateView<DeviceMemorySpace> coords, OutOfBoundsPolicy policy,
  std::shared_ptr<const CoordinateMap> map) const
{
  return Bind(std::move(coords), nullptr, policy, std::move(map));
}

std::shared_ptr<const BoundSource> Geometry::Bind(
  CoordinateView<DeviceMemorySpace> coords,
  std::shared_ptr<const FieldLayout> query_layout, OutOfBoundsPolicy policy,
  std::shared_ptr<const CoordinateMap> map) const
{
  const auto& query_system = coords.GetCoordinateSystem();
  if (map != nullptr) {
    if (!SameCoordinateSystem(query_system, map->GetSourceCoordinateSystem()) ||
        !SameCoordinateSystem(map->GetTargetCoordinateSystem(),
                              GetCoordinateSystem())) {
      throw pcms_error(
        "Geometry::Bind: the query points are in '" +
        std::string(query_system->Kind()) + "' and the map takes '" +
        std::string(map->GetSourceCoordinateSystem()->Kind()) + "' to '" +
        std::string(map->GetTargetCoordinateSystem()->Kind()) +
        "'; it must take the query coordinate system to the geometry's '" +
        std::string(GetCoordinateSystem()->Kind()) + "'");
    }
  } else if (!SameCoordinateSystem(query_system, GetCoordinateSystem())) {
    throw pcms_error("Geometry::Bind: query coordinate system '" +
                     std::string(query_system->Kind()) +
                     "' is not the geometry's coordinate system '" +
                     std::string(GetCoordinateSystem()->Kind()) +
                     "'; pass a CoordinateMap from one to the other as "
                     "Bind's map argument");
  }
  if (policy.mode == OutOfBoundsMode::NEAREST_BOUNDARY &&
      !SupportsNearestBoundary()) {
    throw pcms_error("Geometry::Bind: this geometry does not support "
                     "OutOfBoundsMode::NEAREST_BOUNDARY");
  }
  std::shared_ptr<BoundSource> bound;
  if (map != nullptr) {
    std::shared_ptr<const BoundCoordinateMap> bound_map = map->Bind(coords);
    const CoordinateView<DeviceMemorySpace> mapped = bound_map->MappedPoints();
    bound = BindImpl(BindRequest{mapped, std::move(query_layout), policy,
                                 std::move(bound_map), coords});
  } else {
    bound = BindImpl(
      BindRequest{coords, std::move(query_layout), policy, nullptr, coords});
  }
  PCMS_ALWAYS_ASSERT(bound != nullptr);
  return bound;
}

std::shared_ptr<const BoundSource> Geometry::Bind(
  const ParametricPoints& points, OutOfBoundsPolicy policy) const
{
  if (points.coords.extent(0) != points.ids.extent(0)) {
    throw pcms_error(
      "Geometry::Bind: parametric ids and coords differ in length");
  }
  std::shared_ptr<BoundSource> bound = BindParametricImpl(points, policy);
  PCMS_ALWAYS_ASSERT(bound != nullptr);
  return bound;
}

SupportResults Geometry::Neighbourhood(CoordinateView<DeviceMemorySpace>,
                                       const NeighbourhoodOptions&) const
{
  throw pcms_error("Geometry::Neighbourhood: this geometry has no "
                   "neighbourhood search");
}

DifferentialView Geometry::Differential(const BoundSource&) const
{
  return {};
}

void detail::CheckParametricIds(const ParametricPoints& points, LO num_ids,
                                const std::string& caller,
                                const std::string& entity)
{
  const auto ids = points.ids;
  LO min_id = 0;
  LO max_id = 0;
  Kokkos::parallel_reduce(
    "CheckParametricIds",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0,
                                                            points.NumPoints()),
    KOKKOS_LAMBDA(const LO i, LO& lo, LO& hi) {
      lo = ids(i) < lo ? ids(i) : lo;
      hi = ids(i) > hi ? ids(i) : hi;
    },
    Kokkos::Min<LO>(min_id), Kokkos::Max<LO>(max_id));
  if (points.NumPoints() > 0 && (min_id < 0 || max_id >= num_ids)) {
    throw pcms_error(caller + ": parametric " + entity + " ids span [" +
                     std::to_string(min_id) + ", " + std::to_string(max_id) +
                     "], outside [0, " + std::to_string(num_ids) + ")");
  }
}

} // namespace pcms
