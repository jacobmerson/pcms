#ifndef PCMS_FUNCTION_SPACE_H
#define PCMS_FUNCTION_SPACE_H

#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/coordinate_view.hpp"
#include "field.h"
#include "field_data.h"
#include "field_factory.h"
#include "field_layout.h"
#include "pcms/field/value_view.hpp"
#include "out_of_bounds_policy.h"
#include "point_evaluator.h"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/discretization/discretization.h"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <memory>
#include <optional>
#include <variant>

namespace pcms
{

// FunctionSpace is an abstract interface representing an evaluatable field
// space: layout, evaluation rules, and coordinate interpretation. It produces
// Functions (a Field bound to this space).
//
// Concrete implementations (e.g. LagrangeFunctionSpace) provide backends for
// specific discretizations or mesh types, and their From* factories return a
// shared_ptr. A Function co-owns its space via shared_ptr (obtained through
// enable_shared_from_this), which is why concrete spaces are always
// shared-owned; a stack-allocated FunctionSpace is not constructible (the
// backend constructors are private and reachable only via the shared_ptr
// From* factories).
class FunctionSpace
  : public PointEvaluatorFactory
  , public std::enable_shared_from_this<FunctionSpace>
{
public:
  /// @return the geometry the space is created on: its backend source's, or
  ///         the one given to the constructor; null for a space that does not
  ///         evaluate at points
  [[nodiscard]] std::shared_ptr<const Geometry> GetGeometry()
    const noexcept override
  {
    return evaluator_ != nullptr ? evaluator_->GetGeometry() : geometry_;
  }

  /// @return the geometry's discretization, or the layout's for a space
  ///         without a geometry
  [[nodiscard]] virtual std::shared_ptr<const Discretization>
  GetDiscretization() const noexcept
  {
    const auto geometry = GetGeometry();
    return geometry != nullptr ? geometry->GetDiscretization()
                               : GetLayout()->GetDiscretization();
  }

  virtual std::shared_ptr<const FieldLayout> GetLayout() const noexcept = 0;

  /// @return the geometry's coordinate system, or the layout's for a space
  ///         without a geometry
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept
  {
    const Geometry* geometry =
      evaluator_ != nullptr ? evaluator_->GetGeometry().get() : geometry_.get();
    return geometry != nullptr ? geometry->GetCoordinateSystem()
                               : GetLayout()->GetCoordinateSystem();
  }

  /// @return whether the space's backend source supports `op`; for a space
  ///         without one, whether `op` is Value and the space has a geometry
  ///         (it implements CreatePointEvaluatorImpl), unless it overrides
  ///         this
  [[nodiscard]] bool Supports(DifferentialOperator op) const noexcept override;

  virtual ~FunctionSpace() noexcept = default;

  template <typename T>
  [[nodiscard]] Function<T> CreateFunction(
    std::string name = "", VarianceSignature variance = values::Scalar,
    std::optional<ComponentScaling> scaling = std::nullopt) const;

  template <typename T>
  [[nodiscard]] Function<T> CreateFunction(
    std::string name, VarianceSignature variance,
    std::shared_ptr<const CoordinateSystem> system,
    std::optional<ComponentScaling> scaling = std::nullopt) const;

  // Expert API: wrap externally constructed field data into a Function for this
  // space. The concrete space validates backend-specific field-data type and
  // storage size compatibility.
  template <typename T>
  [[nodiscard]] Function<T> CreateFunction(
    std::string name, std::unique_ptr<FieldData<T>> data) const;

protected:
  /// @param evaluator backend source over this space's geometry, or null for
  ///        a space that implements Supports and CreatePointEvaluatorImpl
  ///        itself and then must pass its geometry
  /// @param geometry the geometry of a space without a backend source, or
  ///        null; both null for a space that does not evaluate at points
  explicit FunctionSpace(
    std::shared_ptr<const PointEvaluatorFactory> evaluator = nullptr,
    std::shared_ptr<const Geometry> geometry = nullptr) noexcept
    : evaluator_(std::move(evaluator)), geometry_(std::move(geometry))
  {
    PCMS_ALWAYS_ASSERT(evaluator_ == nullptr || geometry_ == nullptr);
  }

  /// @brief Creates the evaluator through the backend source.
  /// @throws pcms_error if the space has no backend source
  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    DifferentialOperator op) const override;

  template <typename T>
  static Field<T> WrapField(std::shared_ptr<const FieldLayout> layout,
                            std::unique_ptr<FieldData<T>> data)
  {
    return Field<T>(std::string{}, std::move(layout), std::move(data));
  }

  template <typename T>
  static Function<T> WrapFunction(std::string name,
                                  std::shared_ptr<const FieldLayout> layout,
                                  std::unique_ptr<FieldData<T>> data,
                                  std::shared_ptr<const FunctionSpace> space)
  {
    return Function<T>(std::move(name), std::move(layout), std::move(data),
                       std::move(space));
  }

  virtual FieldVariant CreateFieldImpl(Type storage_type,
                                       ValueBasis basis) const = 0;

  virtual FieldVariant CreateFieldImpl(FieldDataVariant data) const = 0;

private:
  std::shared_ptr<const PointEvaluatorFactory> evaluator_;
  std::shared_ptr<const Geometry> geometry_;
};

template <typename T>
Function<T> FunctionSpace::CreateFunction(
  std::string name, VarianceSignature variance,
  std::optional<ComponentScaling> scaling) const
{
  return CreateFunction<T>(std::move(name), variance, nullptr, scaling);
}

template <typename T>
Function<T> FunctionSpace::CreateFunction(
  std::string name, VarianceSignature variance,
  std::shared_ptr<const CoordinateSystem> system,
  std::optional<ComponentScaling> scaling) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  ValueBasis basis = detail::MakeValueBasis(GetCoordinateSystem(), variance,
                                            std::move(system), scaling);
  detail::ValidateValueSemantics(basis, GetLayout()->GetNumComponents());
  Field<T> field = std::get<Field<T>>(
    CreateFieldImpl(TypeEnumFromType<T>(), std::move(basis)));
  return WrapFunction<T>(std::move(name), std::move(field.layout_),
                         std::move(field.data_), shared_from_this());
}

template <typename T>
Function<T> FunctionSpace::CreateFunction(
  std::string name, std::unique_ptr<FieldData<T>> data) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  if (!data) {
    throw pcms_error("FunctionSpace::CreateFunction: data must not be null");
  }
  detail::ValidateValueSemantics(data->GetValueBasis(),
                                 GetLayout()->GetNumComponents());
  Field<T> field =
    std::get<Field<T>>(CreateFieldImpl(FieldDataVariant{std::move(data)}));
  return WrapFunction<T>(std::move(name), std::move(field.layout_),
                         std::move(field.data_), shared_from_this());
}

} // namespace pcms

#endif // PCMS_FUNCTION_SPACE_H
