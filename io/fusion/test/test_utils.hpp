#pragma once

// The few helpers the Fusion-IO tests need, written against PCMS's installed
// headers (PCMS's own test utilities are not installed).

#include <Kokkos_Core.hpp>
#include <pcms/field/coordinate_system.hpp>
#include <pcms/field/coordinate_view.hpp>
#include <pcms/field/field.h>
#include <pcms/field/value_view.hpp>
#include <pcms/utility/arrays.h>
#include <pcms/utility/memory_spaces.h>
#include <memory>
#include <vector>

namespace pcms::test
{

inline constexpr double ExactTol = 1e-12;
inline constexpr double EvalTol = 1e-10;

/// Device coordinates and a view of them in a coordinate system.
struct DeviceCoordinates
{
  Kokkos::View<Real**, DeviceMemorySpace> view;
  CoordinateView<DeviceMemorySpace> coordinate_view;
};

/// Copies interleaved points (dim values per point) to the device.
inline DeviceCoordinates CreateDeviceCoordinateView(
  const std::vector<Real>& pts,
  std::shared_ptr<const CoordinateSystem> coordinate_system, int dim = 2)
{
  const int n = static_cast<int>(pts.size()) / dim;
  Kokkos::View<Real**, HostMemorySpace> host("host", n, dim);
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < dim; ++c) {
      host(i, c) = pts[static_cast<size_t>(i) * dim + c];
    }
  }
  Kokkos::View<Real**, DeviceMemorySpace> device("device", n, dim);
  DeepCopyMismatchLayouts(device, host);
  return DeviceCoordinates{
    device, CoordinateView<DeviceMemorySpace>{
              ResolveCoordinateSystem(std::move(coordinate_system), dim),
              MakeRank2View(device)}};
}

/// Copies device coordinates (any layout) to a host view.
template <typename CoordView>
inline Kokkos::View<const Real**, HostMemorySpace> CopyCoordinatesToHost(
  const CoordView& coords_device)
{
  auto coords_view = Kokkos::View<Real**, HostMemorySpace>(
    "coords_view", coords_device.extent(0), coords_device.extent(1));
  auto coords_view_device =
    Kokkos::create_mirror_view(DeviceMemorySpace(), coords_view);
  ConvertMismatchLayoutView2D(coords_view_device, coords_device);
  Kokkos::deep_copy(coords_view, coords_view_device);
  return coords_view;
}

/// Tags a device output buffer with a field's stored value basis.
template <typename T>
inline ValueView<T, DeviceMemorySpace> TagLike(
  const Field<T>& field, Kokkos::View<T**, DeviceMemorySpace> values)
{
  return ValueView<T, DeviceMemorySpace>(field.GetData().GetValueBasis(),
                                         MakeRank2View(values));
}

} // namespace pcms::test
