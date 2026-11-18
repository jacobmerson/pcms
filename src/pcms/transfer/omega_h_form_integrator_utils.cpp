#include "pcms/transfer/omega_h_form_integrator_utils.hpp"

namespace pcms::detail
{

Kokkos::View<Real*, DeviceMemorySpace> ElementMeasuresAtPoints(
  const Geometry& geometry, CoordinateView<DeviceMemorySpace> points,
  const OwnerHints& owners, const char* context)
{
  const auto bound = geometry.Bind(points, owners);
  const DifferentialView differential = geometry.Differential(*bound);
  if (differential.Empty()) {
    throw pcms_error(std::string(context) +
                     ": the geometry provides no differential to take element "
                     "measures from");
  }
  PCMS_ALWAYS_ASSERT(owners.entities.extent(0) ==
                     static_cast<size_t>(bound->NumPoints()));
  const int dim = GetMesh(geometry).dim();
  const Real reference_measure = dim == 3 ? 1.0 / 6.0 : 0.5;
  const auto determinant = differential.determinant;
  const bool per_element = differential.per_element;
  const auto element_ids = owners.entities;
  const LO n = bound->NumPoints();
  Kokkos::View<Real*, DeviceMemorySpace> measures(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "element_measures"), n);
  Kokkos::parallel_for(
    "ElementMeasuresAtPoints",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const LO i) {
      // the points lie in their elements, so they are all found and row i of
      // a per-point differential is point i
      const LO row = per_element ? element_ids(i) : i;
      measures(i) = Kokkos::fabs(determinant(row)) * reference_measure;
    });
  return measures;
}

} // namespace pcms::detail
