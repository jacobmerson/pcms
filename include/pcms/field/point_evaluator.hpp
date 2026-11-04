#ifndef PCMS_POINT_EVALUATOR_H
#define PCMS_POINT_EVALUATOR_H

#include "pcms/field/bound_source.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

namespace pcms
{

template <typename T>
class Field;

namespace detail
{

/// @brief Sets every column of the listed rows of `values` to `value`.
/// @param values output view
/// @param rows indices of the rows to set
/// @param value value to write
template <typename T, typename LayoutPolicy>
void FillRows(Rank2View<T, DeviceMemorySpace, LayoutPolicy> values,
              Kokkos::View<const LO*, DeviceMemorySpace> rows, T value)
{
  const auto num_cols = static_cast<int>(values.extent(1));
  Kokkos::parallel_for(
    "PointEvaluator::FillRows",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(rows.extent(0))),
    KOKKOS_LAMBDA(const LO k) {
      for (int c = 0; c < num_cols; ++c) {
        values(rows(k), c) = value;
      }
    });
}

} // namespace detail

/// @brief Evaluates fields at the points of a bound source.
///
/// Created by PointEvaluatorFactory::CreatePointEvaluator. Evaluate writes
/// OutputBasis(stored basis of the field): the stored basis for values, with
/// one covariant index added per derivative order. The output has one row per
/// query point and the field's number of components times the coordinate
/// dimension per added index as columns. EvaluateImpl writes the rows of the
/// bound source's FoundPoints(); in OutOfBoundsMode::FILL, Evaluate then sets
/// the rows of MissingPoints() to the fill value.
/// @tparam T field value type
/// @tparam LayoutPolicy layout of the output view
template <typename T,
          typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class PointEvaluator
{
public:
  /// Writes the field's values at the query points into `values`.
  /// @throws std::invalid_argument if `values` is not shaped
  ///         [NumPoints()][number of output components]
  /// @throws pcms_error if `values` does not claim the basis this evaluator
  ///         writes for the field
  void Evaluate(const Field<T>& field,
                ValueView<T, DeviceMemorySpace, LayoutPolicy> values) const
  {
    CheckOutputShape(field, values);
    detail::CheckWrittenValueBasis(
      "PointEvaluator::Evaluate", values.GetBasis(),
      OutputBasis(field.GetData().GetValueBasis()));
    EvaluateImpl(field, values);
    const auto filled = FilledPoints();
    if (filled.extent(0) > 0) {
      detail::FillRows(values.GetValues(), filled,
                       static_cast<T>(bound_->Policy().fill_value));
    }
  }

  /// @return number of query points
  [[nodiscard]] LO NumPoints() const noexcept { return bound_->NumPoints(); }

  /// @param stored value basis of the evaluated field
  /// @return value basis this evaluator writes: `stored`, with a covariant
  ///         index in the bound points' coordinate system for derivative
  ///         order 1
  [[nodiscard]] virtual ValueBasis OutputBasis(const ValueBasis& stored) const
  {
    if (derivative_order_ == 0) {
      return stored;
    }
    PCMS_ALWAYS_ASSERT(derivative_order_ == 1);
    return FirstDerivativeBasis(stored, bound_->GetCoordinateSystem());
  }

  /// @return indices of the query points whose rows Evaluate sets to the fill
  ///         value; empty unless the policy is OutOfBoundsMode::FILL
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> FilledPoints()
    const noexcept
  {
    if (bound_->Policy().mode != OutOfBoundsMode::FILL) {
      return {};
    }
    return bound_->MissingPoints();
  }

  virtual ~PointEvaluator() noexcept = default;

protected:
  /// @param bound bound source of the query points; not null
  /// @param derivative_order 0 for values, 1 for first derivatives
  explicit PointEvaluator(std::shared_ptr<const BoundSource> bound,
                          int derivative_order = 0)
    : bound_(std::move(bound)), derivative_order_(derivative_order)
  {
  }

  /// @return the bound source of the query points
  [[nodiscard]] const BoundSource& Bound() const noexcept { return *bound_; }

  /// @return 0 for values, 1 for first derivatives
  [[nodiscard]] int DerivativeOrder() const noexcept
  {
    return derivative_order_;
  }

  /// @brief Writes the field's values at the bound source's FoundPoints().
  /// @param field field to evaluate; its output shape and basis are checked
  /// @param values output view; rows of other points may be left unset
  virtual void EvaluateImpl(
    const Field<T>& field,
    ValueView<T, DeviceMemorySpace, LayoutPolicy> values) const = 0;

private:
  void CheckOutputShape(
    const Field<T>& field,
    const ValueView<T, DeviceMemorySpace, LayoutPolicy>& values) const
  {
    const auto num_points = static_cast<size_t>(NumPoints());
    // the field's components times the dimension per index OutputBasis adds
    const ValueBasis stored = field.GetData().GetValueBasis();
    const ValueBasis written = OutputBasis(stored);
    auto num_components =
      static_cast<size_t>(field.GetLayout().GetNumComponents());
    for (int i = stored.Rank(); i < written.Rank(); ++i) {
      num_components *= static_cast<size_t>(written.system->Dimension());
    }
    if (values.extent(0) != num_points || values.extent(1) != num_components) {
      throw std::invalid_argument(
        "PointEvaluator::Evaluate: the output view is shaped [" +
        std::to_string(values.extent(0)) + "][" +
        std::to_string(values.extent(1)) + "] but must be [" +
        std::to_string(num_points) + "][" + std::to_string(num_components) +
        "] (query points x output components)");
    }
  }

  std::shared_ptr<const BoundSource> bound_;
  int derivative_order_;
};

// Variant types using default layout for device memory space
using PointEvaluatorVariant =
  std::variant<std::unique_ptr<PointEvaluator<int8_t>>,
               std::unique_ptr<PointEvaluator<int32_t>>,
               std::unique_ptr<PointEvaluator<int64_t>>,
               std::unique_ptr<PointEvaluator<float>>,
               std::unique_ptr<PointEvaluator<double>>>;

} // namespace pcms

#endif // PCMS_POINT_EVALUATOR_H
