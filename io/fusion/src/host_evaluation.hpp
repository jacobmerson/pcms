#ifndef FUSION_IO_SRC_HOST_EVALUATION_HPP
#define FUSION_IO_SRC_HOST_EVALUATION_HPP

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
#include <vector>

namespace fio::detail
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

/// Bound query points of a host evaluator, with the points and FoundPoints()
/// copied to the host once.
class HostBoundSource : public pcms::BoundSource
{
public:
  /// the bound points, one row per point
  HostRows points;
  /// FoundPoints() on the host
  Kokkos::View<const pcms::LO*, pcms::HostMemorySpace> found;

protected:
  explicit HostBoundSource(const pcms::BindRequest& request)
    : BoundSource(request),
      points("external_points", NumPoints(), PointValues().extent(1)),
      found(Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(),
                                                FoundPoints()))
  {
    pcms::DeepCopyMismatchLayouts(points, PointValues());
  }

  /// Records which points the source evaluates (BoundSource::SetFound).
  /// @param is_found per point, whether the source evaluates it
  void SetFound(const std::vector<char>& is_found)
  {
    Kokkos::View<bool*, pcms::HostMemorySpace> host("external_found",
                                                    is_found.size());
    for (size_t i = 0; i < is_found.size(); ++i) {
      host(i) = is_found[i] != 0;
    }
    BoundSource::SetFound(
      Kokkos::create_mirror_view_and_copy(pcms::DeviceMemorySpace(), host));
    found = Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(),
                                                FoundPoints());
  }
};

/// Host output rows shaped like the evaluator output `out`.
template <typename Output>
HostRows HostRowsLike(const Output& out)
{
  return HostRows("external_values", out.extent(0), out.extent(1));
}

/// Copies host rows to the evaluator output `out`.
template <typename Output>
void CopyToOutput(const HostRows& rows, const Output& out)
{
  Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> device(
    "external_values", rows.extent(0), rows.extent(1));
  pcms::DeepCopyMismatchLayouts(device, rows);
  pcms::CopyDeviceRank2ViewToRank2View(out, pcms::MakeConstRank2View(device));
}

} // namespace fio::detail

#endif // FUSION_IO_SRC_HOST_EVALUATION_HPP
