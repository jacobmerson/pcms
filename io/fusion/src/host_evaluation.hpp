#ifndef FUSION_IO_SRC_HOST_EVALUATION_HPP
#define FUSION_IO_SRC_HOST_EVALUATION_HPP

// Private: what the function spaces wrapping external host evaluators
// (Fusion-IO 1.x, NIMROD's nimeval) share. Their fields hold no values in
// PCMS; each names a quantity of the external source.

#include "pcms/field/coordinate_systems/cylindrical.hpp"
#include "pcms/field/field_data.h"
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

/// The basis an evaluator of `derivative_order` (0 or 1) writes for a field
/// stored in `stored`.
inline pcms::ValueBasis OutputBasisOf(const pcms::ValueBasis& stored,
                                      int derivative_order)
{
  if (derivative_order == 0) {
    return stored;
  }
  return pcms::FirstDerivativeBasis(stored,
                                    pcms::csys::CylindricalRThetaZ::Create());
}

/// Throws unless evaluating `num_components` components at `derivative_order`
/// is supported: Real values, and gradients of single-component quantities.
inline void CheckEvaluatorRequest(std::string_view space, pcms::Type value_type,
                                  int derivative_order, int num_components)
{
  const std::string who(space);
  if (value_type != pcms::Type::Real) {
    throw pcms::pcms_error(who + ": only Real evaluation is supported");
  }
  if (derivative_order > 1) {
    throw pcms::pcms_error(who + ": derivative order " +
                           std::to_string(derivative_order) +
                           " is not supported");
  }
  if (derivative_order == 1 && num_components != 1) {
    throw pcms::pcms_error(who + ": derivatives of quantities with more than "
                                 "one component are not supported yet");
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

/// The query points on the host, each point's coordinates contiguous.
template <typename Coordinates>
HostRows PointsOnHost(const Coordinates& coords)
{
  Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> staged(
    "external_points", coords.extent(0), coords.extent(1));
  pcms::CopyDeviceRank2ViewToRank2View(pcms::MakeRank2View(staged), coords);
  HostRows points("external_points", coords.extent(0), coords.extent(1));
  pcms::DeepCopyMismatchLayouts(points, staged);
  return points;
}

/// Throws unless the evaluator output `out` is [rows][columns].
template <typename Output>
void CheckOutputShape(std::string_view space, const Output& out, size_t rows,
                      size_t columns)
{
  if (out.extent(0) != rows || out.extent(1) != columns) {
    throw pcms::pcms_error(std::string(space) + ": output is shaped [" +
                           std::to_string(out.extent(0)) + "][" +
                           std::to_string(out.extent(1)) + "] but expected [" +
                           std::to_string(rows) + "][" +
                           std::to_string(columns) + "]");
  }
}

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
