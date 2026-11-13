#ifndef PCMS_FIELD_POINT_EVALUATOR_FACTORY_HPP
#define PCMS_FIELD_POINT_EVALUATOR_FACTORY_HPP

#include "pcms/field/bound_source.hpp"
#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/coordinate_view.hpp"
#include "pcms/field/evaluation_request.h"
#include "pcms/field/field_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <memory>
#include <string>
#include <variant>

namespace pcms
{

class FunctionSpace;

/// @brief Source of point evaluators.
///
/// Bind computes the per-point data for a set of query points once;
/// CreatePointEvaluator creates evaluators of any value type and derivative
/// order from the result.
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
  /// @param derivative_order 0 for values, 1 for first derivatives
  /// @return evaluator that writes OutputBasis of the field's stored basis
  /// @throws pcms_error if `bound` is null or was bound by another source, or
  ///         the source cannot evaluate `derivative_order`
  template <typename T>
  [[nodiscard]] std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    std::shared_ptr<const BoundSource> bound, int derivative_order = 0) const;

  /// @brief Binds to the request's points and creates an evaluator.
  /// @tparam T field value type
  /// @param request query points and out-of-bounds policy
  /// @return evaluator of the field values
  template <typename T>
  [[deprecated("bind with Bind(coords | space, policy) and pass the bound "
               "source")]] [[nodiscard]] std::unique_ptr<PointEvaluator<T>>
  CreatePointEvaluator(const EvaluationRequest& request) const;

  /// @return the source whose Bind this source uses and whose bound sources
  ///         it accepts: itself, unless it wraps another source
  [[nodiscard]] virtual const PointEvaluatorFactory& BindingSource()
    const noexcept
  {
    return *this;
  }

  /// @return whether this source can bind with
  ///         OutOfBoundsMode::NEAREST_BOUNDARY
  [[nodiscard]] virtual bool SupportsNearestBoundary() const noexcept
  {
    return false;
  }

  virtual ~PointEvaluatorFactory() noexcept = default;

protected:
  /// @brief Computes the per-point data for a request.
  /// @param request query points in GetCoordinateSystem(); may be released
  ///        after the call
  /// @return bound source holding copies of everything evaluators need
  virtual std::shared_ptr<BoundSource> BindImpl(
    const EvaluationRequest& request) const = 0;

  /// @brief Creates an evaluator from a bound source.
  /// @param value_type field value type
  /// @param bound result of BindImpl on this source
  /// @param derivative_order 0 for values, 1 for first derivatives
  /// @return evaluator for `value_type`
  /// @throws pcms_error for an unsupported value type or derivative order
  virtual PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    int derivative_order) const = 0;

private:
  std::shared_ptr<const BoundSource> BindRequest(
    const EvaluationRequest& request) const;
  void CheckBoundByThis(const BoundSource& bound) const;
};

/// @brief Source that evaluates fields of one value type from bound sources
///        of one type.
///
/// Implements BindImpl and CreatePointEvaluatorImpl: CreatePointEvaluator
/// accepts only bound sources this source created, so the bound source is
/// passed on as a `Bound`.
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

protected:
  /// @param system coordinate system of the query points
  explicit TypedPointEvaluatorFactory(
    std::shared_ptr<const CoordinateSystem> system)
    : system_(std::move(system))
  {
  }

  /// @brief Computes the per-point data for a request.
  /// @param request query points in GetCoordinateSystem(); may be released
  ///        after the call
  /// @return the bound source
  virtual std::shared_ptr<Bound> BindPoints(
    const EvaluationRequest& request) const = 0;

  /// @brief Creates an evaluator from a bound source of this source.
  /// @param bound result of BindPoints on this source
  /// @param derivative_order 0 for values, 1 for first derivatives
  /// @return the evaluator
  /// @throws pcms_error for an unsupported derivative order
  virtual std::unique_ptr<PointEvaluator<T>> CreateEvaluator(
    std::shared_ptr<const Bound> bound, int derivative_order) const = 0;

private:
  std::shared_ptr<BoundSource> BindImpl(
    const EvaluationRequest& request) const final
  {
    return BindPoints(request);
  }

  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    int derivative_order) const final
  {
    if (value_type != TypeEnumFromType<T>()) {
      throw pcms_error("CreatePointEvaluator: this source only supports "
                       "fields of its own value type");
    }
    return CreateEvaluator(std::static_pointer_cast<const Bound>(bound),
                           derivative_order);
  }

  std::shared_ptr<const CoordinateSystem> system_;
};

template <typename T>
std::unique_ptr<PointEvaluator<T>> PointEvaluatorFactory::CreatePointEvaluator(
  std::shared_ptr<const BoundSource> bound, int derivative_order) const
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  if (bound == nullptr) {
    throw pcms_error("CreatePointEvaluator: bound source must not be null");
  }
  CheckBoundByThis(*bound);
  if (derivative_order < 0) {
    throw pcms_error("CreatePointEvaluator: derivative order must be "
                     "non-negative, got " +
                     std::to_string(derivative_order));
  }
  return std::get<std::unique_ptr<PointEvaluator<T>>>(CreatePointEvaluatorImpl(
    TypeEnumFromType<T>(), std::move(bound), derivative_order));
}

template <typename T>
std::unique_ptr<PointEvaluator<T>> PointEvaluatorFactory::CreatePointEvaluator(
  const EvaluationRequest& request) const
{
  return CreatePointEvaluator<T>(BindRequest(request));
}

} // namespace pcms

#endif // PCMS_FIELD_POINT_EVALUATOR_FACTORY_HPP
