#ifndef PCMS_POINT_EVALUATOR_H
#define PCMS_POINT_EVALUATOR_H

#include "pcms/field/value_view.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>

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
//   [num_query_points][num_components]
// The caller must provide the full output buffer. Successful Evaluate calls
// fill the entire buffer.
//
// The write is tagged: an evaluator produces OutputBasis(stored) for a field
// stored in basis `stored` -- the stored basis itself for values, and one more
// covariant index for derivatives -- so the ValueView the caller supplies must
// claim that basis (all rank-0 bases compare equal, so scalar callers may tag
// with ValueBasis{}). Implementations gate on it with CheckEvaluateWriteTag
// and throw pcms_error on mismatch.
template <typename T,
          typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class PointEvaluator
{
public:
  virtual void Evaluate(
    const Field<T>& field,
    ValueView<T, DeviceMemorySpace, LayoutPolicy> values) const = 0;

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
  /// Throws pcms_error unless the output view claims the basis this evaluator
  /// writes for the source field, OutputBasis of its stored basis.
  void CheckEvaluateWriteTag(
    const char* context, const Field<T>& field,
    const ValueView<T, DeviceMemorySpace, LayoutPolicy>& values) const
  {
    detail::CheckWrittenValueBasis(
      context, values.GetBasis(), OutputBasis(field.GetData().GetValueBasis()));
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
