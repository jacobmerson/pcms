#include "pcms/field/evaluator/mls_point_cloud.h"

namespace pcms::detail
{

Kokkos::View<bool*, DeviceMemorySpace> HasSupports(
  const Omega_h::LOs& supports_ptr, LO n)
{
  Kokkos::View<bool*, DeviceMemorySpace> found(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "mls_found"), n);
  detail::ForEach(
    "MLSBinding::found", n, KOKKOS_LAMBDA(const LO i) {
      found(i) = supports_ptr[i + 1] > supports_ptr[i];
    });
  return found;
}

} // namespace pcms::detail
