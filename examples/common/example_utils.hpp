#ifndef PCMS_EXAMPLES_COMMON_EXAMPLE_UTILS_HPP
#define PCMS_EXAMPLES_COMMON_EXAMPLE_UTILS_HPP

// Helpers shared by the C++ examples. They use the public PCMS API only and
// do two small jobs:
//
//   * move small matrices between host memory and the device views that
//     PCMS's evaluation functions take (on a CPU-only build both are host
//     memory, and the copies are cheap), and
//   * tally the checks an example asserts, so that every example prints what
//     it verified and returns a nonzero exit code if a check failed.

#include <pcms/field/field.hpp>
#include <pcms/field/value_view.hpp>
#include <pcms/utility/arrays.hpp>
#include <pcms/utility/coordinate_view.hpp>
#include <pcms/utility/memory_spaces.hpp>
#include <pcms/utility/types.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace example
{

using HostMatrix = Kokkos::View<pcms::Real**, pcms::HostMemorySpace>;
using DeviceMatrix = Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace>;

/// A device matrix with the rows of a host matrix.
inline DeviceMatrix ToDevice(const HostMatrix& host)
{
  return Kokkos::create_mirror_view_and_copy(pcms::DeviceMemorySpace(), host);
}

/// A device matrix with the rows of `flat`, `width` numbers per row.
inline DeviceMatrix ToDevice(const std::vector<pcms::Real>& flat, int width)
{
  const std::size_t rows = flat.size() / static_cast<std::size_t>(width);
  HostMatrix host("host", rows, width);
  for (std::size_t i = 0; i < rows; ++i) {
    for (int d = 0; d < width; ++d) {
      host(i, d) = flat[i * width + d];
    }
  }
  return ToDevice(host);
}

/// A host copy of a device matrix.
inline HostMatrix ToHost(const DeviceMatrix& device)
{
  return Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), device);
}

/// A host copy of the matrix a PCMS view refers to (the coordinates of the
/// DOF holders of a layout, for example).
inline HostMatrix ToHost(
  pcms::Rank2View<const pcms::Real, pcms::DeviceMemorySpace> view)
{
  const std::size_t cols = view.extent(1);
  DeviceMatrix device("copy", view.extent(0), cols);
  Kokkos::parallel_for(
    "example::ToHost",
    Kokkos::RangePolicy<pcms::DeviceMemorySpace::execution_space>(
      0, view.extent(0)),
    KOKKOS_LAMBDA(const int i) {
      for (std::size_t j = 0; j < cols; ++j) {
        device(i, j) = view(i, j);
      }
    });
  return ToHost(device);
}

/// Query points: the rows of `points`, in `system`.
inline pcms::CoordinateView<pcms::DeviceMemorySpace> Points(
  std::shared_ptr<const pcms::CoordinateSystem> system,
  const DeviceMatrix& points)
{
  return pcms::CoordinateView<pcms::DeviceMemorySpace>(
    std::move(system), pcms::MakeConstRank2View(points));
}

/// An output buffer for an evaluation: `values` tagged with `basis`, which
/// must be the basis the evaluation writes (that of the field for values).
inline pcms::ValueView<pcms::Real, pcms::DeviceMemorySpace> Output(
  pcms::ValueBasis basis, DeviceMatrix& values)
{
  return pcms::ValueView<pcms::Real, pcms::DeviceMemorySpace>(
    std::move(basis), pcms::Rank2View<pcms::Real, pcms::DeviceMemorySpace>(
                        values.data(), values.extent(0), values.extent(1)));
}

/// The coordinates of the DOF holders of a field, one row per holder.
inline HostMatrix DofCoordinates(const pcms::Field<pcms::Real>& field)
{
  return ToHost(field.GetLayout().GetDOFHolderCoordinates().GetValues());
}

/// Sets the values of the DOF holders of a field from a host matrix with one
/// row per holder and one column per component.
inline void SetDofData(pcms::Field<pcms::Real>& field, const HostMatrix& data)
{
  field.SetDOFHolderDataHost(
    pcms::ValueView<const pcms::Real, pcms::HostMemorySpace>(
      field.GetData().GetValueBasis(), pcms::MakeConstRank2View(data)));
}

/// The values of the DOF holders of a field as a host matrix.
inline HostMatrix DofData(const pcms::Field<pcms::Real>& field)
{
  const auto values = field.GetDOFHolderDataHost().GetValues();
  HostMatrix copy("dof_data", values.extent(0), values.extent(1));
  for (std::size_t i = 0; i < values.extent(0); ++i) {
    for (std::size_t j = 0; j < values.extent(1); ++j) {
      copy(i, j) = values(i, j);
    }
  }
  return copy;
}

/// Tallies the checks of an example. `Check` prints one line per check;
/// `Finish` prints the verdict and returns the exit code of the program.
class Report
{
public:
  /// Checks that `error` (a size, an error norm) is at most `tolerance`.
  void CheckAtMost(const std::string& what, pcms::Real error,
                   pcms::Real tolerance)
  {
    Record(what, error <= tolerance, "%.3e <= %.1e", error, tolerance);
  }

  /// Checks a statement, with the number it is about.
  void Check(const std::string& what, bool holds, long value)
  {
    Record(what, holds, "%ld", value);
  }

  [[nodiscard]] int Finish() const
  {
    if (failures_ == 0) {
      std::printf("SUCCESS: %d checks passed\n", checks_);
      return 0;
    }
    std::printf("FAILED: %d of %d checks failed\n", failures_, checks_);
    return 1;
  }

private:
  template <typename... Args>
  void Record(const std::string& what, bool ok, const char* format,
              Args... args)
  {
    char detail[128];
    std::snprintf(detail, sizeof(detail), format, args...);
    std::printf("  [%s] %s: %s\n", ok ? "ok" : "FAIL", what.c_str(), detail);
    ++checks_;
    failures_ += ok ? 0 : 1;
  }

  int checks_ = 0;
  int failures_ = 0;
};

} // namespace example

#endif // PCMS_EXAMPLES_COMMON_EXAMPLE_UTILS_HPP
