#ifndef PCMS_GEOMETRY_PARAMETRIC_POINTS_HPP
#define PCMS_GEOMETRY_PARAMETRIC_POINTS_HPP

#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <string>

namespace pcms
{

/// @brief Points given in a geometry's own parametric coordinates, so no
///        search is needed to bind them.
///
/// The meaning of `ids` and `coords` is defined by the geometry kind. For a
/// mesh geometry, `ids(i)` is the element of point i and `coords(i, :)` its
/// reference coordinates (barycentric, Dim + 1 columns). For a uniform grid,
/// `ids(i)` is the cell and `coords(i, :)` the position in the cell in [0, 1].
struct ParametricPoints
{
  Kokkos::View<const LO*, DeviceMemorySpace> ids;
  Kokkos::View<const Real**, DeviceMemorySpace> coords;

  [[nodiscard]] LO NumPoints() const noexcept
  {
    return static_cast<LO>(ids.extent(0));
  }
};

namespace detail
{

/// @brief Checks that every id of `points` indexes one of `num_ids` entities.
/// @param points parametric points to check
/// @param num_ids number of entities the ids index
/// @param caller name of the checking function, for the message
/// @param entity what the ids index, e.g. "element", for the message
/// @throws pcms_error naming the range of the ids if one is outside
///         [0, num_ids)
void CheckParametricIds(const ParametricPoints& points, LO num_ids,
                        const std::string& caller, const std::string& entity);

} // namespace detail

} // namespace pcms

#endif // PCMS_GEOMETRY_PARAMETRIC_POINTS_HPP
