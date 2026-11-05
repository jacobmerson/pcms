#ifndef FUSION_IO_TEST_TEST_UTILS_HPP
#define FUSION_IO_TEST_TEST_UTILS_HPP

// The few helpers the Fusion-IO tests need, written against PCMS's installed
// headers (PCMS's own test utilities are not installed).

#include <Kokkos_Core.hpp>
#include <pcms/field/coordinate_system.hpp>
#include <pcms/field/coordinate_view.hpp>
#include <pcms/field/field.h>
#include <pcms/field/function_space.h>
#include <pcms/field/value_view.hpp>
#include <pcms/utility/arrays.h>
#include <pcms/utility/memory_spaces.h>
#include <memory>
#include <vector>

namespace fio::test
{

inline constexpr double ExactTol = 1e-12;
inline constexpr double EvalTol = 1e-10;

using pcms::DeviceMemorySpace;
using pcms::HostMemorySpace;
using pcms::Real;

/// Device coordinates and a view of them in a coordinate system.
struct DeviceCoordinates
{
  Kokkos::View<Real**, DeviceMemorySpace> view;
  pcms::CoordinateView<DeviceMemorySpace> coordinate_view;
};

/// Copies interleaved points (dim values per point) to the device.
inline DeviceCoordinates CreateDeviceCoordinateView(
  const std::vector<Real>& pts,
  std::shared_ptr<const pcms::CoordinateSystem> coordinate_system, int dim = 2)
{
  const int n = static_cast<int>(pts.size()) / dim;
  Kokkos::View<Real**, HostMemorySpace> host("host", n, dim);
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < dim; ++c) {
      host(i, c) = pts[static_cast<size_t>(i) * dim + c];
    }
  }
  Kokkos::View<Real**, DeviceMemorySpace> device("device", n, dim);
  pcms::DeepCopyMismatchLayouts(device, host);
  return DeviceCoordinates{
    device, pcms::CoordinateView<DeviceMemorySpace>{
              pcms::ResolveCoordinateSystem(std::move(coordinate_system), dim),
              pcms::MakeRank2View(device)}};
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
  pcms::ConvertMismatchLayoutView2D(coords_view_device, coords_device);
  Kokkos::deep_copy(coords_view, coords_view_device);
  return coords_view;
}

/// Tags a device output buffer with a field's stored value basis.
template <typename T>
inline pcms::ValueView<T, DeviceMemorySpace> TagLike(
  const pcms::Field<T>& field, Kokkos::View<T**, DeviceMemorySpace> values)
{
  return pcms::ValueView<T, DeviceMemorySpace>(field.GetData().GetValueBasis(),
                                               pcms::MakeRank2View(values));
}

/// Values or, for derivative_order 1, gradients of `field` evaluated through
/// `space` at `points` ((R, phi, Z) row-major), on the host.
inline Kokkos::View<Real**, HostMemorySpace> EvaluateAt(
  const pcms::FunctionSpace& space, const pcms::Field<Real>& field,
  const std::vector<Real>& points, int derivative_order = 0)
{
  const auto coords =
    CreateDeviceCoordinateView(points, space.GetCoordinateSystem(), 3);
  const auto evaluator = space.CreatePointEvaluator<Real>(
    space.Bind(coords.coordinate_view), derivative_order);
  const auto basis = evaluator->OutputBasis(field.GetData().GetValueBasis());
  const int columns =
    derivative_order == 0 ? field.GetLayout().GetNumComponents() : 3;
  Kokkos::View<Real**, DeviceMemorySpace> out("out", points.size() / 3,
                                              columns);
  evaluator->Evaluate(field, pcms::ValueView<Real, DeviceMemorySpace>(
                               basis, pcms::MakeRank2View(out)));
  return Kokkos::create_mirror_view_and_copy(HostMemorySpace(), out);
}

/// Gradient of the scalar `field` at `points`: (d/dR, (1/R) d/dphi, d/dZ)
/// per row.
inline Kokkos::View<Real**, HostMemorySpace> EvaluateGradientAt(
  const pcms::FunctionSpace& space, const pcms::Field<Real>& field,
  const std::vector<Real>& points)
{
  return EvaluateAt(space, field, points, 1);
}

/// Values of `function` at `points`, evaluated through its own space.
inline Kokkos::View<Real**, HostMemorySpace> EvaluateAt(
  const pcms::Function<Real>& function, const std::vector<Real>& points)
{
  return EvaluateAt(function.GetSpace(), function, points);
}

} // namespace fio::test

#endif // FUSION_IO_TEST_TEST_UTILS_HPP
