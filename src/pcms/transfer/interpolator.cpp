#include "pcms/transfer/interpolator.h"

namespace pcms
{

namespace detail
{

bool SameDOFHolderSites(const FieldLayout& a, const FieldLayout& b)
{
  if (&a == &b) {
    return true;
  }
  const auto ca = a.GetDOFHolderCoordinates();
  const auto cb = b.GetDOFHolderCoordinates();
  if (!SameCoordinateSystem(ca.GetCoordinateSystem(),
                            cb.GetCoordinateSystem())) {
    return false;
  }
  const auto va = ca.GetValues();
  const auto vb = cb.GetValues();
  if (va.extent(0) != vb.extent(0) || va.extent(1) != vb.extent(1)) {
    return false;
  }
  if (va.data_handle() == vb.data_handle()) {
    return true;
  }
  const auto num_dims = static_cast<int>(va.extent(1));
  LO num_different = 0;
  Kokkos::parallel_reduce(
    "same_dof_holder_sites",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(va.extent(0))),
    KOKKOS_LAMBDA(const LO i, LO& count) {
      for (int d = 0; d < num_dims; ++d) {
        if (va(i, d) != vb(i, d)) {
          ++count;
          return;
        }
      }
    },
    num_different);
  return num_different == 0;
}

} // namespace detail

} // namespace pcms
