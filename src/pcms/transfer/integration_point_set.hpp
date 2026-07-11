#ifndef PCMS_TRANSFER_INTEGRATION_POINT_SET_H
#define PCMS_TRANSFER_INTEGRATION_POINT_SET_H

#include "pcms/field/coordinate_system.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/types.h"

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <utility>

namespace pcms
{

// An ordered collection of global coordinates at which a source field must
// be evaluated for integration. The ordering is significant: sampled values
// passed back to a LinearFormIntegrator must use the same point order as
// stored here.
//
// This type intentionally does not expose quadrature weights, basis values,
// entity ids, or any other assembly metadata. Those remain private to the
// concrete integrator implementation. It owns the coordinate buffer; callers
// receive a non-owning CoordinateView from GetCoordinates().
template <typename MemorySpace>
class IntegrationPointSet
{
public:
  IntegrationPointSet(CoordinateSystem coordinate_system,
                      Kokkos::View<Real**, MemorySpace> coords)
    : coordinate_system_(coordinate_system), coords_(std::move(coords))
  {
  }

  // Global coordinates of all integration points, shape [num_points][dim].
  CoordinateView<MemorySpace> GetCoordinates() const
  {
    return CoordinateView<MemorySpace>(coordinate_system_,
                                       MakeConstRank2View(coords_));
  }

  // Number of integration points.
  [[nodiscard]]
  std::size_t NumPoints() const
  {
    return coords_.extent(0);
  }

private:
  CoordinateSystem coordinate_system_;
  Kokkos::View<Real**, MemorySpace> coords_;
};

} // namespace pcms

#endif // PCMS_TRANSFER_INTEGRATION_POINT_SET_H
