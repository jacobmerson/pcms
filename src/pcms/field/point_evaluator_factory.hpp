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

  /// Shorthand for CreatePointEvaluator<T>(Bind(request)).
  template <typename T>
  [[nodiscard]] std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    const EvaluationRequest& request) const;

  /// @return the source whose Bind this source uses and whose bound sources
  ///         it accepts: itself, unless it wraps another source
  [[nodiscard]] virtual const PointEvaluatorFactory& BindingSource()
    const noexcept
  {
    return *this;
  }

  virtual ~PointEvaluatorFactory() noexcept = default;

protected:
  /// Performs the per-point work for `request`, whose points are in
  /// GetCoordinateSystem(). The default defers the search to
  /// CreatePointEvaluatorImpl(Type, const EvaluationRequest&).
  virtual std::shared_ptr<BoundSource> BindImpl(
    const EvaluationRequest& request) const;

  /// Creates an evaluator from a bound source produced by BindImpl. The default
  /// serves the deferred bound source of the default BindImpl, for values
  /// only.
  virtual PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    int derivative_order) const;

  /// Searches and creates an evaluator in one step. Implemented by backends
  /// that do not implement BindImpl.
  virtual PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, const EvaluationRequest& request) const;

private:
  std::shared_ptr<const BoundSource> BindRequest(
    const EvaluationRequest& request) const;
  void CheckBoundByThis(const BoundSource& bound) const;
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
