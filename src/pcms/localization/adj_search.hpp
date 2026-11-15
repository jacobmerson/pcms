#ifndef PCMS_LOCALIZATION_ADJ_SEARCH_HPP
#define PCMS_LOCALIZATION_ADJ_SEARCH_HPP

#include <pcms/geometry/supports.hpp>
#include <pcms/localization/mls_support_helpers.h>
#include <pcms/utility/memory_spaces.h>
#include <pcms/utility/mesh_geometry.h>
#include <pcms/utility/print.h>
#include <pcms/utility/types.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace pcms
{

/// @brief Supports of query points among the vertices of `source_mesh`: a
///        breadth-first search over the vertex adjacency from the vertices of
///        each point's start element, keeping the vertices within the cutoff.
/// @param target_coords query points, flattened [x0, y0, ..., x1, y1, ...]
/// @param start_elements the element of `source_mesh` each query point's
///        search starts from; a point with -1 gets no supports
/// @param cutoffDistance initial squared radius of every query point
SupportResults searchNeighbors(
  Omega_h::Mesh& source_mesh, const Omega_h::Reals& target_coords,
  Kokkos::View<const LO*, DeviceMemorySpace> start_elements,
  Omega_h::Real& cutoffDistance, Omega_h::LO min_req_support = 12,
  Omega_h::LO max_allowed_support = 36, bool adapt_radius = true);

/// @brief Supports of the vertices of a 2D mesh among its element centroids,
///        by a breadth-first search over the elements around each vertex.
SupportResults searchNeighbors(Omega_h::Mesh& mesh,
                               Omega_h::Real cutoffDistance,
                               Omega_h::LO min_support = 12,
                               bool adapt_radius = true);
} // namespace pcms

#endif // PCMS_LOCALIZATION_ADJ_SEARCH_HPP
