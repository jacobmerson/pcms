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

// PointEvaluator<T> is bound to a specific set of query points. Created by
// PointEvaluatorFactory::CreatePointEvaluator from a bound source, which holds
// the backend-specific localization work (spatial search, barycentric
// coordinate computation, etc.) done once by Bind and shared by every
// evaluator created from it. Evaluate may then be called repeatedly for
// different FieldData objects at zero additional localization cost.
//
// Field compatibility with a PointEvaluator is a precondition of Evaluate.
// Implementations should check this with a cheap backend-specific identity
// check rather than deep structural comparison and should throw pcms_error on
// mismatch.
//
// Evaluate writes results into a tagged rank-2 output view with shape:
//   [NumPoints()][num_components of the field x dim per derivative]
// The caller must provide the full output buffer. Successful Evaluate calls
// fill the entire buffer.
//
// The write is tagged: an evaluator produces OutputBasis(stored) for a field
// stored in basis `stored` -- the stored basis itself for values, and one more
// covariant index for derivatives -- so the ValueView the caller supplies must
// claim that basis (all rank-0 bases compare equal, so scalar callers may tag
// with ValueBasis{}). Each index OutputBasis adds multiplies the number of
// output columns by the dimension of its coordinate system.
//
// Evaluate checks the output's shape and tag before calling EvaluateImpl, so
// implementations may assume both.
template <typename T,
          typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class PointEvaluator
{
public:
  /// Writes the field's values at the query points into `values`.
  /// @throws std::invalid_argument if `values` is not shaped
  ///         [NumPoints()][number of output components] (see above)
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
    // the field's components, times the dimension for each index the output
    // basis adds (one per derivative)
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
