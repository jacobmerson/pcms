#ifndef PCMS_FIELD_POINT_EVALUATOR_FACTORY_HPP
#define PCMS_FIELD_POINT_EVALUATOR_FACTORY_HPP

#include "pcms/field/bound_source.hpp"
#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/coordinate_view.hpp"
#include "pcms/field/differential_operator.hpp"
#include "pcms/field/field_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <variant>

namespace pcms
{

class FunctionSpace;
class PointEvaluatorFactory;

namespace detail
{
/// @brief Internal: the source that binds for `source`, for wrappers and for
///        PCMS code that reuses a backend's search.
/// @return `source`, unless it binds through a source it wraps
[[nodiscard]] const PointEvaluatorFactory& BindingSourceOf(
  const PointEvaluatorFactory& source) noexcept;
} // namespace detail

/// @brief Source of point evaluators.
///
/// Bind computes the per-point data for a set of query points once;
/// CreatePointEvaluator creates evaluators of any value type and supported
/// operator from the result.
class PointEvaluatorFactory
{
public:
  /// @return coordinate system of the query points passed to Bind
  [[nodiscard]] virtual const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const = 0;

  /// @brief Binds this source to query points.
  /// @param coords query points in GetCoordinateSystem(); may be released
  ///        after the call
  /// @param policy out-of-bounds policy of the evaluators created from the
  ///        result
  /// @return bound source for CreatePointEvaluator
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    CoordinateView<DeviceMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const;

  /// @brief Binds this source to the DOF-holder coordinates of a space.
  /// @param target space whose DOF holders are the query points
  /// @param policy out-of-bounds policy of the evaluators created from the
  ///        result
  /// @return bound source for CreatePointEvaluator
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    const FunctionSpace& target, OutOfBoundsPolicy policy = {}) const;

  /// @brief Creates an evaluator at the points of a bound source.
  /// @tparam T field value type
  /// @param bound result of Bind on this source
  /// @param op the operator the evaluator applies
  /// @return evaluator that writes OutputBasis of the field's stored basis
  /// @throws pcms_error if `bound` is null or was bound by another source, or
  ///         the source does not support `op`
  template <typename T>
  [[nodiscard]] std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    std::shared_ptr<const BoundSource> bound,
    DifferentialOperator op = DifferentialOperator::Value) const;

  /// @return whether CreatePointEvaluator accepts `op`
  [[nodiscard]] virtual bool Supports(
    DifferentialOperator op) const noexcept = 0;

  virtual ~PointEvaluatorFactory() noexcept = default;

protected:
  /// @return the source whose Bind this source uses and whose bound sources
  ///         it accepts: itself, unless it wraps another source
  [[nodiscard]] virtual const PointEvaluatorFactory& BindingSource()
    const noexcept
  {
    return *this;
  }

  /// @brief Computes the per-point data for a request. Not called for
  ///        sources that bind through another BindingSource().
  /// @param request query points in GetCoordinateSystem(); may be released
  ///        after the call
  /// @return bound source holding copies of everything evaluators need
  /// @throws pcms_error unless overridden
  virtual std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const;

  /// @brief Creates an evaluator from a bound source.
  /// @param value_type field value type
  /// @param bound result of BindImpl on this source
  /// @param op the operator the evaluator applies; Supports(op) is true
  /// @return evaluator for `value_type`
  /// @throws pcms_error for an unsupported value type
  virtual PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    DifferentialOperator op) const = 0;

private:
  friend const PointEvaluatorFactory& detail::BindingSourceOf(
    const PointEvaluatorFactory& source) noexcept;

  std::shared_ptr<const BoundSource> BindThrough(
    const BindRequest& request) const;
  void CheckBoundByThis(const BoundSource& bound) const;
};

/// @brief Source that evaluates fields of one value type from bound sources
///        of one type.
///
/// Implements BindImpl and CreatePointEvaluatorImpl: BindImpl checks the
/// out-of-bounds mode against SupportsNearestBoundary(), and
/// CreatePointEvaluatorImpl checks the value type and passes the bound source
/// on as a `Bound` (CreatePointEvaluator accepts only bound sources this
/// source created). Supports reports the operators given to the constructor.
/// @tparam T value type of the evaluated fields
/// @tparam Bound bound-source type BindPoints creates
template <typename T, typename Bound>
class TypedPointEvaluatorFactory : public PointEvaluatorFactory
{
public:
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const final
  {
    return system_;
  }

  [[nodiscard]] bool Supports(DifferentialOperator op) const noexcept final
  {
    return supported_.count(op) != 0;
  }

protected:
  /// @return whether BindPoints supports OutOfBoundsMode::NEAREST_BOUNDARY;
  ///         BindImpl rejects that mode otherwise
  [[nodiscard]] virtual bool SupportsNearestBoundary() const noexcept
  {
    return false;
  }

  /// @param system coordinate system of the query points
  /// @param supported operators CreateEvaluator implements; Value is always
  ///        included
  explicit TypedPointEvaluatorFactory(
    std::shared_ptr<const CoordinateSystem> system,
    std::initializer_list<DifferentialOperator> supported = {})
    : system_(std::move(system)), supported_{DifferentialOperator::Value}
  {
    supported_.insert(supported.begin(), supported.end());
  }

  /// @brief Computes the per-point data for a request.
  /// @param request query points in GetCoordinateSystem(); may be released
  ///        after the call
  /// @return the bound source
  virtual std::shared_ptr<Bound> BindPoints(
    const BindRequest& request) const = 0;

  /// @brief Creates an evaluator from a bound source of this source.
  /// @param bound result of BindPoints on this source
  /// @param op one of the operators given to the constructor
  /// @return the evaluator
  virtual std::unique_ptr<PointEvaluator<T>> CreateEvaluator(
    std::shared_ptr<const Bound> bound, DifferentialOperator op) const = 0;

private:
  std::shared_ptr<BoundSource> BindImpl(const BindRequest& request) const final
  {
    if (request.policy.mode == OutOfBoundsMode::NEAREST_BOUNDARY &&
        !this->SupportsNearestBoundary()) {
      throw pcms_error("Bind: this source does not support "
                       "OutOfBoundsMode::NEAREST_BOUNDARY");
    }
    return BindPoints(request);
  }

  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    DifferentialOperator op) const final
  {
    if (value_type != TypeEnumFromType<T>()) {
      throw pcms_error("CreatePointEvaluator: this source only supports "
                       "fields of its own value type");
    }
    return CreateEvaluator(std::static_pointer_cast<const Bound>(bound), op);
  }

  std::shared_ptr<const CoordinateSystem> system_;
  std::set<DifferentialOperator> supported_;
};

template <typename T>
std::unique_ptr<PointEvaluator<T>> PointEvaluatorFactory::CreatePointEvaluator(
  std::shared_ptr<const BoundSource> bound, DifferentialOperator op) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  if (bound == nullptr) {
    throw pcms_error("CreatePointEvaluator: bound source must not be null");
  }
  CheckBoundByThis(*bound);
  if (!Supports(op)) {
    throw pcms_error("CreatePointEvaluator: " + std::string(ToString(op)) +
                     " is not supported by this source");
  }
  return std::get<std::unique_ptr<PointEvaluator<T>>>(
    CreatePointEvaluatorImpl(TypeEnumFromType<T>(), std::move(bound), op));
}

} // namespace pcms

#endif // PCMS_FIELD_POINT_EVALUATOR_FACTORY_HPP
