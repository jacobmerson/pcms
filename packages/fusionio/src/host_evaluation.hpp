#ifndef FUSIONIO_SRC_HOST_EVALUATION_HPP
#define FUSIONIO_SRC_HOST_EVALUATION_HPP

// Private: what the function spaces wrapping external host evaluators
// (Fusion-IO 1.x, NIMROD's nimeval) share. Their fields hold no values in
// PCMS; each names a quantity of the external source.

#include "pcms/field/coordinate_systems/cylindrical.hpp"
#include "pcms/field/field_data.h"
#include "pcms/field/bound_source.hpp"
#include "pcms/field/function_space.h"
#include "pcms/field/value_view.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace fusionio::detail
{

using HostRows =
  Kokkos::View<pcms::Real**, Kokkos::LayoutRight, pcms::HostMemorySpace>;

/// A quantity of an external source as PCMS field data. PCMS holds none of
/// its values, so it provides no DOFAccess.
template <typename Quantity>
class ExternalFieldData final : public pcms::FieldData<pcms::Real>
{
public:
  ExternalFieldData(Quantity quantity, pcms::ValueBasis basis)
    : quantity(std::move(quantity)), basis_(std::move(basis))
  {
  }

  [[nodiscard]] pcms::FieldValueType GetValueType() const override
  {
    return pcms::ValueTypeOfRank(basis_.Rank());
  }

  [[nodiscard]] const pcms::ValueBasis& GetValueBasis() const override
  {
    return basis_;
  }

  Quantity quantity;

private:
  pcms::ValueBasis basis_;
};

/// The basis of a quantity's values: physical cylindrical components for a
/// vector, none for scalars.
inline pcms::ValueBasis QuantityBasis(bool vector)
{
  if (!vector) {
    return {};
  }
  return {pcms::csys::CylindricalRThetaZ::Create(),
          pcms::ComponentScaling::Physical, pcms::values::Vector};
}

/// Throws unless a gradient (derivative order 1) of a quantity with
/// `num_components` components is supported: only single-component quantities
/// have gradients so far.
inline void CheckGradientComponents(std::string_view space,
                                    int derivative_order, int num_components)
{
  if (derivative_order == 1 && num_components != 1) {
    throw pcms::pcms_error(std::string(space) +
                           ": derivatives of quantities with more than one "
                           "component are not supported yet");
  }
}

/// Throws that `quantity` is not one of `available`.
[[noreturn]] inline void ThrowUnknownQuantity(
  std::string_view space, std::string_view quantity,
  const std::vector<std::string>& available)
{
  std::string list;
  for (const auto& name : available) {
    list += (list.empty() ? "" : ", ") + name;
  }
  throw pcms::pcms_error(
    std::string(space) + ": \"" + std::string(quantity) +
    "\" is not a quantity of this source; it has: " + list);
}

/// Throws unless a quantity's component count is the space's.
inline void CheckComponents(std::string_view space, std::string_view quantity,
                            int components, int space_components)
{
  if (components != space_components) {
    throw pcms::pcms_error(std::string(space) + ": \"" + std::string(quantity) +
                           "\" has " + std::to_string(components) +
                           " components but the space holds " +
                           std::to_string(space_components));
  }
}

/// Host views of query points, one row per point.
using HostPoints =
  Kokkos::View<const pcms::Real**, Kokkos::LayoutRight, pcms::HostMemorySpace>;

/// Device rows on the host in row-major layout: `rows` itself where device
/// memory is host memory and row-major, otherwise a copy.
inline HostPoints HostRowMajor(
  const Kokkos::View<const pcms::Real**, pcms::DeviceMemorySpace>& rows)
{
  using Rows = Kokkos::View<const pcms::Real**, pcms::DeviceMemorySpace>;
  if constexpr (std::is_same_v<pcms::DeviceMemorySpace,
                               pcms::HostMemorySpace> &&
                std::is_same_v<Rows::array_layout, Kokkos::LayoutRight>) {
    return rows;
  } else {
    HostRows copy(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "external_points"),
      rows.extent(0), rows.extent(1));
    pcms::DeepCopyMismatchLayouts(copy, rows);
    return copy;
  }
}

/// Bound query points of a host evaluator, with the points and FoundPoints()
/// on the host (views of the bound source's own where the host can read
/// them, so binding copies nothing more on host-only builds).
class HostBoundSource : public pcms::BoundSource
{
public:
  /// the bound points, one row per point
  HostPoints points;
  /// FoundPoints() on the host
  Kokkos::View<const pcms::LO*, pcms::HostMemorySpace> found;

protected:
  explicit HostBoundSource(const pcms::BindRequest& request)
    : BoundSource(request),
      points(HostRowMajor(PointValues())),
      found(Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(),
                                                FoundPoints()))
  {
  }

  /// Records which points the source evaluates (BoundSource::SetFound).
  /// Every point stays found if `is_found` holds no false.
  /// @param is_found per point, whether the source evaluates it
  void SetFound(Kokkos::View<const bool*, pcms::HostMemorySpace> is_found)
  {
    bool all_found = true;
    for (size_t i = 0; i < is_found.extent(0) && all_found; ++i) {
      all_found = is_found(i);
    }
    if (all_found) {
      return;
    }
    BoundSource::SetFound(
      Kokkos::create_mirror_view_and_copy(pcms::DeviceMemorySpace(), is_found));
    found = Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(),
                                                FoundPoints());
  }
};

/// Whether an evaluator output is host memory in row-major layout, so that
/// host evaluators can write its rows directly.
template <typename Output>
inline constexpr bool HostWritable =
  std::is_same_v<typename Output::accessor_type::memory_space,
                 pcms::HostMemorySpace> &&
  std::is_same_v<typename Output::layout_type, Kokkos::layout_right>;

/// Host output rows shaped like the evaluator output `out`: a view of `out`
/// itself where the host can write it (HostWritable), otherwise scratch rows
/// for CopyToOutput.
template <typename Output>
HostRows HostRowsLike(const Output& out)
{
  if constexpr (HostWritable<Output>) {
    // unmanaged: out's owner keeps the memory
    return HostRows(out.data_handle(), out.extent(0), out.extent(1));
  } else {
    return HostRows("external_values", out.extent(0), out.extent(1));
  }
}

/// Copies host rows from HostRowsLike(out) to the evaluator output `out`;
/// nothing to do if they are `out` itself.
template <typename Output>
void CopyToOutput(const HostRows& rows, const Output& out)
{
  if constexpr (!HostWritable<Output>) {
    Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> device(
      "external_values", rows.extent(0), rows.extent(1));
    pcms::DeepCopyMismatchLayouts(device, rows);
    pcms::CopyDeviceRank2ViewToRank2View(out, pcms::MakeConstRank2View(device));
  }
}

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_HOST_EVALUATION_HPP
