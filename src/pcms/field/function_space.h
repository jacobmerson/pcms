#ifndef PCMS_FUNCTION_SPACE_H
#define PCMS_FUNCTION_SPACE_H

#include "coordinate_system.h"
#include "evaluation_request.h"
#include "field.h"
#include "field_data.h"
#include "field_evaluator_factory.h"
#include "field_layout.h"
#include "field_metadata.h"
#include "out_of_bounds_policy.h"
#include "point_evaluator.h"
#include "pcms/discretization/discretization.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <memory>
#include <variant>

namespace pcms
{

namespace detail
{

inline size_t ExpectedFlatFieldDataSize(const FieldLayout& layout)
{
  return static_cast<size_t>(layout.GetNumOwnedDofHolder()) *
         static_cast<size_t>(layout.GetNumComponents());
}

} // namespace detail

// Compile-time gate: true only for the five supported field value types.
template <typename T>
inline constexpr bool is_supported_field_type_v =
  std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t> ||
  std::is_same_v<T, int64_t> || std::is_same_v<T, float> ||
  std::is_same_v<T, double>;

// FunctionSpace is an abstract interface representing an evaluatable field
// space: layout, evaluation rules, and coordinate interpretation.
//
// Concrete implementations (e.g. LagrangeFunctionSpace) provide backends for
// specific discretizations or mesh types.
//
// FunctionSpace is used as the parameter type for operation objects such as
// Interpolator<T>, so that operations are not coupled to a specific backend.
//
// FunctionSpace inherits enable_shared_from_this because CreateFunction stamps
// a shared_ptr<const FunctionSpace> into each Function it produces. Concrete
// spaces must therefore be owned by a shared_ptr (the From* factories return
// one); calling CreateFunction on a stack-allocated space throws bad_weak_ptr.
class FunctionSpace : public std::enable_shared_from_this<FunctionSpace>
{
public:
  virtual std::shared_ptr<const Discretization> GetDiscretization()
    const noexcept
  {
    return GetLayout()->GetDiscretization();
  }

  virtual std::shared_ptr<const FieldLayout> GetLayout() const noexcept = 0;

  virtual CoordinateSystem GetCoordinateSystem() const noexcept = 0;

  virtual ~FunctionSpace() noexcept = default;

  // Create a new field with freshly allocated data for this function space.
  // Compile-time error for unsupported T; runtime error for T unsupported by
  // the concrete backend.
  template <typename T>
  [[nodiscard]] Field<T> CreateField(FieldMetadata metadata = {}) const;

  // Expert API: wrap externally constructed field data into a Field for this
  // function space. The concrete function space validates backend-specific
  // field-data type and storage size compatibility.
  template <typename T>
  [[nodiscard]] Field<T> CreateField(std::unique_ptr<FieldData<T>> data) const;

  // Like CreateField, but returns a Function that retains a shared reference to
  // this space (so it can be evaluated / used to build transfer operators).
  // Requires this space to be owned by a shared_ptr (see class note).
  template <typename T>
  [[nodiscard]] Function<T> CreateFunction(FieldMetadata metadata = {}) const;

  template <typename T>
  [[nodiscard]] Function<T> CreateFunction(
    std::unique_ptr<FieldData<T>> data) const;

  // Create a point evaluator for the given evaluation request.
  // Compile-time error for unsupported T; runtime error for T or capability
  // unsupported by the concrete backend.
  //
  // EvaluationRequest is a construction-time object: it supplies the query
  // coordinates, out-of-bounds policy, and any optional provenance that may
  // help the backend choose an optimized localization path. The resulting
  // PointEvaluator caches only the resolved state needed for repeated
  // Evaluate(...) calls; it is not required to retain the original request.
  template <typename T>
  [[nodiscard]] std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    const EvaluationRequest& request) const;

protected:
  template <typename T>
  static Field<T> WrapField(std::shared_ptr<const FieldLayout> layout,
                            std::unique_ptr<FieldData<T>> data)
  {
    return Field<T>(std::move(layout), std::move(data));
  }

  template <typename T>
  static Function<T> WrapFunction(std::shared_ptr<const FieldLayout> layout,
                                  std::unique_ptr<FieldData<T>> data,
                                  std::shared_ptr<const FunctionSpace> space)
  {
    return Function<T>(std::move(layout), std::move(data), std::move(space));
  }

  virtual FieldVariant CreateFieldImpl(Type value_type,
                                       FieldMetadata metadata) const = 0;

  virtual FieldVariant CreateFieldImpl(FieldDataVariant data) const = 0;

  virtual PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, const EvaluationRequest& request) const = 0;
};

// Wrap a by-value FunctionSpace (typically the result of a From* factory) in a
// shared_ptr so it can produce Functions and participate in transfers via
// AddFunction/CreateTransfer. Ordinary by-value use — CreateField, point
// evaluation, constructing a transfer operator directly — does not need this;
// only Function (which retains a shared_ptr<const FunctionSpace>) does.
//
//   auto space = pcms::Share(LagrangeFunctionSpace::FromMesh(...));
//   auto fn = space->CreateFunction<Real>();
template <typename Space>
[[nodiscard]] std::shared_ptr<Space> Share(Space&& space)
{
  static_assert(
    std::is_base_of_v<FunctionSpace, std::remove_reference_t<Space>>,
    "Share expects a FunctionSpace-derived type");
  return std::make_shared<std::remove_reference_t<Space>>(
    std::forward<Space>(space));
}

template <typename T>
Field<T> FunctionSpace::CreateField(FieldMetadata metadata) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  return std::get<Field<T>>(CreateFieldImpl(TypeEnumFromType<T>(), metadata));
}

template <typename T>
Field<T> FunctionSpace::CreateField(std::unique_ptr<FieldData<T>> data) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  if (!data) {
    throw pcms_error("FunctionSpace::CreateField: data must not be null");
  }
  return std::get<Field<T>>(CreateFieldImpl(FieldDataVariant{std::move(data)}));
}

template <typename T>
Function<T> FunctionSpace::CreateFunction(FieldMetadata metadata) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  Field<T> field = CreateField<T>(metadata);
  return WrapFunction<T>(std::move(field.layout_), std::move(field.data_),
                         shared_from_this());
}

template <typename T>
Function<T> FunctionSpace::CreateFunction(
  std::unique_ptr<FieldData<T>> data) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  Field<T> field = CreateField<T>(std::move(data));
  return WrapFunction<T>(std::move(field.layout_), std::move(field.data_),
                         shared_from_this());
}

template <typename T>
std::unique_ptr<PointEvaluator<T>> FunctionSpace::CreatePointEvaluator(
  const EvaluationRequest& request) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  return std::get<std::unique_ptr<PointEvaluator<T>>>(
    CreatePointEvaluatorImpl(TypeEnumFromType<T>(), request));
}

inline EvaluationRequest EvaluationRequest::FromCoordinates(
  CoordinateView<DeviceMemorySpace> coords, OutOfBoundsPolicy policy)
{
  return EvaluationRequest(coords, nullptr, policy);
}

inline EvaluationRequest EvaluationRequest::FromLayout(
  std::shared_ptr<const FieldLayout> layout, OutOfBoundsPolicy policy)
{
  if (layout == nullptr) {
    throw pcms_error("EvaluationRequest::FromLayout: layout must not be null");
  }
  // Must evaluate GetDOFHolderCoordinates() before std::move(layout)
  auto coords = layout->GetDOFHolderCoordinates();
  return EvaluationRequest(coords, std::move(layout), policy);
}

inline EvaluationRequest EvaluationRequest::FromFunctionSpace(
  const FunctionSpace& function_space, OutOfBoundsPolicy policy)
{
  return FromLayout(function_space.GetLayout(), policy);
}

} // namespace pcms

#endif // PCMS_FUNCTION_SPACE_H
