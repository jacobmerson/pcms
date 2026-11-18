#include "pcms/field/evaluator/uniform_grid_spline.h"

namespace pcms::detail
{

void GatherPoints(Kokkos::View<const Real**, DeviceMemorySpace> points,
                  Kokkos::View<const LO*, DeviceMemorySpace> rows,
                  Kokkos::View<Real*, DeviceMemorySpace> x,
                  Kokkos::View<Real*, DeviceMemorySpace> y)
{
  Kokkos::parallel_for(
    "gather_spline_points",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(rows.extent(0))),
    KOKKOS_LAMBDA(const LO k) {
      x(k) = points(rows(k), 0);
      y(k) = points(rows(k), 1);
    });
}

} // namespace pcms::detail
