#ifndef FUSIONIO_TEST_TEST_UTILS_HPP
#define FUSIONIO_TEST_TEST_UTILS_HPP

// The few helpers the Fusion-IO tests need, written against PCMS's installed
// headers (PCMS's own test utilities are not installed).

#include <Kokkos_Core.hpp>
#include <pcms/utility/coordinate_system.hpp>
#include <pcms/utility/coordinate_view.hpp>
#include <pcms/field/field.hpp>
#include <pcms/field/function_space.hpp>
#include <pcms/field/value_view.hpp>
#include <pcms/geometry/geometry.hpp>
#include <pcms/utility/arrays.hpp>
#include <pcms/utility/memory_spaces.hpp>
#include <memory>
#include <vector>

namespace fusionio::test
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
              std::move(coordinate_system), pcms::MakeRank2View(device)}};
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

/// Binds `points` (row-major, the geometry's dimension per point) on
/// `geometry`.
inline std::shared_ptr<const pcms::BoundPoints> Bind(
  const pcms::Geometry& geometry, const std::vector<Real>& points,
  pcms::OutOfBoundsPolicy policy = {})
{
  const int dim = geometry.GetCoordinateSystem()->Dimension();
  const auto coords =
    CreateDeviceCoordinateView(points, geometry.GetCoordinateSystem(), dim);
  return geometry.Bind(coords.coordinate_view, policy);
}

/// `op` of `field` evaluated by `evaluator`, on the host: one row per point.
inline Kokkos::View<Real**, HostMemorySpace> EvaluateWith(
  const pcms::PointEvaluator<Real>& evaluator, const pcms::Field<Real>& field)
{
  const auto stored = field.GetData().GetValueBasis();
  const auto basis = evaluator.OutputBasis(stored);
  int columns = field.GetLayout().GetNumComponents();
  for (int i = stored.Rank(); i < basis.Rank(); ++i) {
    columns *= basis.system->Dimension();
  }
  Kokkos::View<Real**, DeviceMemorySpace> out("out", evaluator.NumPoints(),
                                              columns);
  evaluator.Evaluate(field, pcms::ValueView<Real, DeviceMemorySpace>(
                              basis, pcms::MakeRank2View(out)));
  // the device layout may differ from the host's (LayoutLeft on CUDA)
  Kokkos::View<Real**, HostMemorySpace> host("host", out.extent(0),
                                             out.extent(1));
  pcms::DeepCopyMismatchLayouts(host, out);
  return host;
}

/// `op` of `function` at `points` (row-major, in its geometry's coordinates),
/// evaluated through its own space, on the host.
inline Kokkos::View<Real**, HostMemorySpace> EvaluateAt(
  const pcms::Function<Real>& function, const std::vector<Real>& points,
  pcms::DifferentialOperator op = pcms::DifferentialOperator::Value)
{
  const auto& space = function.GetSpace();
  const auto evaluator =
    space.CreatePointEvaluator<Real>(Bind(*space.GetGeometry(), points), op);
  return EvaluateWith(*evaluator, function);
}

/// Gradient of the scalar `function` at `points`: (d/dR, (1/R) d/dphi, d/dZ)
/// per row on a cylindrical (R, phi, Z) geometry.
inline Kokkos::View<Real**, HostMemorySpace> EvaluateGradientAt(
  const pcms::Function<Real>& function, const std::vector<Real>& points)
{
  return EvaluateAt(function, points, pcms::DifferentialOperator::Gradient);
}

} // namespace fusionio::test

#endif // FUSIONIO_TEST_TEST_UTILS_HPP
