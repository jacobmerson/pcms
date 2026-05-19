#ifndef PCMS_TRANSFER_INTEGRATION_POINT_SET_H
#define PCMS_TRANSFER_INTEGRATION_POINT_SET_H

#include "pcms/field/coordinate_system.h"

namespace pcms
{

// An ordered collection of global coordinates at which a source field must
// be evaluated for integration. The ordering is significant: sampled values
// passed back to a LinearFormIntegrator must use the same point order as
// returned here.
//
// This interface intentionally does not expose quadrature weights, basis
// values, entity ids, or any other assembly metadata. Those remain private
// to the concrete integrator implementation.
template <typename MemorySpace>
class IntegrationPointSet
{
public:
  // Global coordinates of all integration points, shape [num_points][dim].
  virtual CoordinateView<MemorySpace> GetCoordinates() const = 0;

  virtual ~IntegrationPointSet() noexcept = default;
};

} // namespace pcms

#endif // PCMS_TRANSFER_INTEGRATION_POINT_SET_H
