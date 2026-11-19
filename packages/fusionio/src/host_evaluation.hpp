#ifndef FUSIONIO_SRC_HOST_EVALUATION_HPP
#define FUSIONIO_SRC_HOST_EVALUATION_HPP

// Private: what the function spaces wrapping external host evaluators
// (Fusion-IO 1.x, NIMROD's nimeval) share. Their fields hold no values in
// PCMS; each names a quantity of the external source.

#include "pcms/field/differential_operator.hpp"
#include "pcms/field/field_data.hpp"
#include "pcms/field/function_space.hpp"
#include "pcms/field/function_spaces.hpp"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/bound_points.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/cylindrical.hpp"
#include <Kokkos_Core.hpp>
#include <functional>
#include <memory>
#include <set>
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
    return basis_.Rank() == 0 ? pcms::FieldValueType::Scalar
                              : pcms::FieldValueType::Vector;
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
  return {pcms::csys::CreateCylindricalRThetaZ(),
          pcms::ComponentScaling::Physical, pcms::values::Vector};
}

/// Throws unless `op` of a quantity with `num_components` components is
/// supported: only single-component quantities have gradients so far.
inline void CheckGradientComponents(std::string_view space,
                                    pcms::DifferentialOperator op,
                                    int num_components)
{
  if (op == pcms::DifferentialOperator::Gradient && num_components != 1) {
    throw pcms::pcms_error(std::string(space) +
                           ": derivatives of quantities with more than one "
                           "component are not supported yet");
  }
}

/// Throws that `quantity` is not one of `available`.
[[noreturn]] inline void ThrowUnknownQuantity(
  std::string_view source, std::string_view quantity,
  const std::vector<std::string>& available)
{
  std::string list;
  for (const auto& name : available) {
    list += (list.empty() ? "" : ", ") + name;
  }
  throw pcms::pcms_error(
    std::string(source) + ": \"" + std::string(quantity) +
    "\" is not a quantity of this source; it has: " + list);
}

/// @brief A space whose fields name a quantity of an external source:
///        creating a field from a value type and basis throws.
class ExternalSpace final : public pcms::FunctionSpace
{
public:
  using pcms::FunctionSpace::FunctionSpace;

protected:
  [[nodiscard]] pcms::FieldDataVariant CreateFieldData(
    pcms::Type /*storage_type*/, pcms::ValueBasis /*basis*/) const override
  {
    throw pcms::pcms_error("fusionio: a value type does not name a quantity; "
                           "use Source::CreateFunction(quantity)");
  }
};

/// Host views of query points, one row per point.
using HostPoints =
  Kokkos::View<const pcms::Real**, Kokkos::LayoutRight, pcms::HostMemorySpace>;

/// Device rows on the host in row-major layout: an unmanaged view of `rows`
/// itself where device memory is host memory and row-major, otherwise a copy.
/// The view of `rows` is unmanaged so that it does not hold the storage of the
/// BoundPoints `rows` belongs to, which Geometry::Rebind recycles only when
/// nothing else holds it.
inline HostPoints HostRowMajor(
  const Kokkos::View<const pcms::Real**, pcms::DeviceMemorySpace>& rows)
{
  using Rows = Kokkos::View<const pcms::Real**, pcms::DeviceMemorySpace>;
  if constexpr (std::is_same_v<pcms::DeviceMemorySpace,
                               pcms::HostMemorySpace> &&
                std::is_same_v<Rows::array_layout, Kokkos::LayoutRight>) {
    return HostPoints(rows.data(), rows.extent(0), rows.extent(1));
  } else {
    HostRows copy(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "external_points"),
      rows.extent(0), rows.extent(1));
    pcms::DeepCopyMismatchLayouts(copy, rows);
    return copy;
  }
}

/// Device indices on the host: an unmanaged view of `indices` itself where
/// device memory is host memory (see HostRowMajor), otherwise a copy.
inline Kokkos::View<const pcms::LO*, pcms::HostMemorySpace> HostIndices(
  const Kokkos::View<const pcms::LO*, pcms::DeviceMemorySpace>& indices)
{
  if constexpr (std::is_same_v<pcms::DeviceMemorySpace,
                               pcms::HostMemorySpace>) {
    return Kokkos::View<const pcms::LO*, pcms::HostMemorySpace>(
      indices.data(), indices.extent(0));
  } else {
    return Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(),
                                               indices);
  }
}

/// Bound query points of a host evaluator, with LocatedPointValues() and
/// FoundPoints() on the host (unmanaged views of the bound source's own
/// arrays where the host can read them, so binding copies nothing more on
/// host-only builds).
class HostBoundPoints : public pcms::BoundPoints
{
public:
  /// LocatedPointValues() on the host, one row per point
  HostPoints points;
  /// FoundPoints() on the host
  Kokkos::View<const pcms::LO*, pcms::HostMemorySpace> found;

protected:
  HostBoundPoints(std::shared_ptr<const pcms::Geometry> geometry,
                  const pcms::BindRequest& request)
    : BoundPoints(std::move(geometry), request)
  {
    RefreshHostViews();
  }

  /// Points the host views at the current points, after a rebind has reset
  /// them (BoundPoints::Reset).
  void RefreshHostViews()
  {
    points = HostRowMajor(LocatedPointValues());
    found = HostIndices(FoundPoints());
  }

  /// Records which points the source evaluates (BoundPoints::SetFound).
  /// Every point stays found if `is_found` holds no false.
  /// @param is_found per point, whether the source evaluates it
  /// @throws pcms_error in OutOfBoundsMode::ERROR if a point is not found
  void SetFound(Kokkos::View<const bool*, pcms::HostMemorySpace> is_found)
  {
    bool all_found = true;
    for (size_t i = 0; i < is_found.extent(0) && all_found; ++i) {
      all_found = is_found(i);
    }
    if (all_found) {
      return;
    }
    BoundPoints::SetFound(
      Kokkos::create_mirror_view_and_copy(pcms::DeviceMemorySpace(), is_found));
    found = HostIndices(FoundPoints());
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
    // element-wise, whatever the layout of `out`
    const size_t num_components = rows.extent(1);
    Kokkos::parallel_for(
      "fusionio::CopyToOutput",
      Kokkos::RangePolicy<pcms::DeviceMemorySpace::execution_space>(
        0, rows.extent(0)),
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t c = 0; c < num_components; ++c) {
          out(i, c) = device(i, c);
        }
      });
  }
}

/// Creates the evaluators of the quantities of one representation of an
/// external source from BoundPoints of kind `Bound` (a local version of
/// PCMS's internal TypedPointEvaluatorFactory).
template <typename Bound>
class ExternalEvaluatorFactory final : public pcms::PointEvaluatorFactory
{
public:
  using Create =
    std::function<std::unique_ptr<pcms::PointEvaluator<pcms::Real>>(
      std::shared_ptr<const Bound>, pcms::DifferentialOperator)>;

  /// @param name for error messages
  /// @param supported the operators `create` implements besides Value
  ExternalEvaluatorFactory(std::string name,
                           std::shared_ptr<const pcms::Geometry> geometry,
                           std::set<pcms::DifferentialOperator> supported,
                           Create create)
    : name_(std::move(name)),
      geometry_(std::move(geometry)),
      supported_(std::move(supported)),
      create_(std::move(create))
  {
    supported_.insert(pcms::DifferentialOperator::Value);
  }

  [[nodiscard]] std::shared_ptr<const pcms::Geometry> GetGeometry()
    const noexcept override
  {
    return geometry_;
  }

  [[nodiscard]] bool Supports(
    pcms::DifferentialOperator op) const noexcept override
  {
    return supported_.contains(op);
  }

protected:
  // PointEvaluatorFactory::CreatePointEvaluator has checked the geometry and
  // the operator.
  pcms::PointEvaluatorVariant CreatePointEvaluatorImpl(
    pcms::Type value_type, std::shared_ptr<const pcms::BoundPoints> bound,
    pcms::DifferentialOperator op) const override
  {
    if (value_type != pcms::Type::Real) {
      throw pcms::pcms_error(name_ + ": only Real fields are evaluated");
    }
    auto typed = std::dynamic_pointer_cast<const Bound>(std::move(bound));
    if (typed == nullptr) {
      throw pcms::pcms_error(name_ + ": BoundPoints of an unexpected kind");
    }
    return create_(std::move(typed), op);
  }

private:
  std::string name_;
  std::shared_ptr<const pcms::Geometry> geometry_;
  std::set<pcms::DifferentialOperator> supported_;
  Create create_;
};

/// The space of the quantities of one representation of an external source
/// on `geometry`: `num_components` components, evaluated by the evaluators
/// `create(bound, op, layout)` returns for the BoundPoints `bound` of kind
/// `Bound` (see ExternalEvaluatorFactory). PCMS holds none of the values: its
/// functions name quantities (ExternalFieldData).
template <typename Bound, typename CreateEvaluator>
std::shared_ptr<const pcms::FunctionSpace> CreateExternalSpace(
  std::string name, std::shared_ptr<const pcms::Geometry> geometry,
  int num_components, std::set<pcms::DifferentialOperator> supported,
  CreateEvaluator create)
{
  std::shared_ptr<const pcms::FieldLayout> layout =
    pcms::CreateEmptyLayout(geometry->GetCoordinateSystem(), num_components);
  auto evaluator = std::make_shared<ExternalEvaluatorFactory<Bound>>(
    std::move(name), geometry, std::move(supported),
    [layout, create = std::move(create)](std::shared_ptr<const Bound> bound,
                                         pcms::DifferentialOperator op) {
      return std::unique_ptr<pcms::PointEvaluator<pcms::Real>>(
        create(std::move(bound), op, layout));
    });
  return std::make_shared<ExternalSpace>(
    std::move(layout), std::move(evaluator), std::move(geometry));
}

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_HOST_EVALUATION_HPP
