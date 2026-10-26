#ifndef PCMS_POINT_EVALUATOR_H
#define PCMS_POINT_EVALUATOR_H

#include "pcms/field/value_view.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>

namespace pcms
{

template <typename T>
class Field;

/// @brief Evaluates fields at a fixed set of query points.
///
/// Created by PointEvaluatorFactory::CreatePointEvaluator from a bound source.
/// Evaluate writes OutputBasis(stored basis of the field): the stored basis for
/// values, with one covariant index added per derivative order. The output has
/// one row per query point and the field's number of components times the
/// coordinate dimension per added index as columns. Implementations of
/// EvaluateImpl may assume the output's shape and basis are checked, and throw
/// pcms_error for an incompatible field.
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
  }

  /// Number of query points, fixed at construction.
  [[nodiscard]] virtual LO NumPoints() const = 0;

  /// Value basis this evaluator writes for a field stored in `stored`.
  [[nodiscard]] virtual ValueBasis OutputBasis(const ValueBasis& stored) const
  {
    return stored;
  }

  /// Query points whose output rows Evaluate sets to the out-of-bounds fill
  /// value rather than to field values; empty unless the evaluator was created
  /// with OutOfBoundsMode::FILL. Fixed at construction.
  [[nodiscard]] virtual Kokkos::View<const LO*, DeviceMemorySpace>
  FilledPoints() const = 0;

  virtual ~PointEvaluator() noexcept = default;

protected:
  /// Evaluate after its checks: `values` is shaped and tagged correctly.
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
