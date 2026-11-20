#ifndef PCMS_GEOMETRY_DETAIL_GEOMETRY_HPP
#define PCMS_GEOMETRY_DETAIL_GEOMETRY_HPP

#include "pcms/geometry/geometry.hpp"
#include <string>

namespace pcms::detail
{

/// @brief The checks every Neighbourhood makes of its arguments.
/// @param geometry geometry whose Neighbourhood is called
/// @param coords the query points
/// @param options the search options
/// @param caller name of the calling function, for the message
/// @throws pcms_error if `coords` is not in the geometry's coordinate system,
///         that system does not have the identity metric, or
///         `options.max_supports` is neither 0 nor 3 * `options.min_supports`
void CheckNeighbourhoodArguments(
  const Geometry& geometry, const CoordinateView<DeviceMemorySpace>& coords,
  const NeighbourhoodOptions& options, const std::string& caller);

/// @brief Checks that every id of a hint is one of `num_entities` entities.
/// @param ids entity ids, e.g. OwnerHints::entities
/// @param num_entities number of entities the ids index
/// @param caller name of the checking function, for the message
/// @param entity what the ids index, e.g. "element", for the message
/// @throws pcms_error naming the range of the ids if one is outside
///         [0, num_entities)
void CheckEntityIds(Kokkos::View<const LO*, DeviceMemorySpace> ids,
                    LO num_entities, const std::string& caller,
                    const std::string& entity);

/// @brief Checks that the weights of a quadrature over `geometry` are
///        measures: they carry |det J| only, which is the physical measure in
///        Cartesian coordinates alone.
/// @param geometry the geometry the quadrature integrates over
/// @param caller name of the calling function, for the message
/// @throws pcms_error if the geometry's coordinate system is not Cartesian
// Until coordinate systems provide a metric (S5)
void CheckCartesianMeasure(const Geometry& geometry, const std::string& caller);

} // namespace pcms::detail

#endif // PCMS_GEOMETRY_DETAIL_GEOMETRY_HPP
