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

// Evaluation happens in two steps. Bind performs every per-point computation
// for a query point set (the point search) once; CreatePointEvaluator then
// creates evaluators from the bound source cheaply, one per value type and
// derivative order, all sharing that search. This mirrors CoordinateMap::Bind.
class PointEvaluatorFactory
{
public:
  /// Coordinate system that query points handed to Bind must be expressed in.
  [[nodiscard]] virtual const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const = 0;

  /// Binds this source to query points.
  /// @param coords query points in GetCoordinateSystem(); copied as needed, so
  ///        they may be released on return
  /// @param policy out-of-bounds policy of every evaluator created from the
  ///        result
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    CoordinateView<DeviceMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const;

  /// Binds this source to the DOF-holder sites of `target`. The layout is kept
  /// as provenance, which some backends localize faster from.
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    const FunctionSpace& target, OutOfBoundsPolicy policy = {}) const;

  /// Creates an evaluator over the points `bound` was bound to, without
  /// searching again.
  /// @param bound result of Bind on this factory
  /// @param derivative_order 0 for values, 1 for first derivatives; throws for
  ///        an order this source cannot evaluate
  template <typename T>
  [[nodiscard]] std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    std::shared_ptr<const BoundSource> bound, int derivative_order = 0) const;

  /// Shorthand for CreatePointEvaluator<T>(Bind(request)).
  template <typename T>
  [[nodiscard]] std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    const EvaluationRequest& request) const;

  /// The factory that performs this factory's binds, whose bound sources this
  /// factory accepts: itself, unless it wraps another source and evaluates over
  /// that source's search.
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
