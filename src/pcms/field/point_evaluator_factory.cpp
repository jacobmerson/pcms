#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/function_space.h"
#include <string>

namespace pcms
{

std::shared_ptr<const BoundSource> PointEvaluatorFactory::Bind(
  CoordinateView<DeviceMemorySpace> coords, OutOfBoundsPolicy policy) const
{
  return BindRequest(EvaluationRequest::FromCoordinates(coords, policy));
}

std::shared_ptr<const BoundSource> PointEvaluatorFactory::Bind(
  const FunctionSpace& target, OutOfBoundsPolicy policy) const
{
  return BindRequest(EvaluationRequest::FromFunctionSpace(target, policy));
}

std::shared_ptr<const BoundSource> PointEvaluatorFactory::BindRequest(
  const EvaluationRequest& request) const
{
  const PointEvaluatorFactory& binder = BindingSource();
  if (&binder != this) {
    return binder.BindRequest(request);
  }
  if (!SameCoordinateSystem(request.coords.GetCoordinateSystem(),
                            GetCoordinateSystem())) {
    throw pcms_error(
      "Bind: query coordinate system '" +
      std::string(request.coords.GetCoordinateSystem()->Kind()) +
      "' is not the factory's coordinate system '" +
      std::string(GetCoordinateSystem()->Kind()) +
      "'; map the points with a CoordinateMap or evaluate through a "
      "TransformedSource");
  }
  std::shared_ptr<BoundSource> bound = BindImpl(request);
  PCMS_ALWAYS_ASSERT(bound != nullptr);
  bound->bound_by_ = this;
  return bound;
}

void PointEvaluatorFactory::CheckBoundByThis(const BoundSource& bound) const
{
  if (bound.bound_by_ != &BindingSource()) {
    throw pcms_error("CreatePointEvaluator: the bound source was bound by "
                     "another factory; bind this factory to the points");
  }
}

std::shared_ptr<BoundSource> PointEvaluatorFactory::BindImpl(
  const EvaluationRequest& request) const
{
  return std::make_shared<detail::DeferredBoundSource>(request);
}

PointEvaluatorVariant PointEvaluatorFactory::CreatePointEvaluatorImpl(
  Type value_type, std::shared_ptr<const BoundSource> bound,
  int derivative_order) const
{
  if (derivative_order != 0) {
    throw pcms_error("CreatePointEvaluator: this source does not evaluate "
                     "derivatives (requested order " +
                     std::to_string(derivative_order) + ")");
  }
  const auto& deferred = detail::BoundSourceAs<detail::DeferredBoundSource>(
    *bound, "PointEvaluatorFactory::CreatePointEvaluatorImpl");
  return CreatePointEvaluatorImpl(value_type, deferred.Request());
}

PointEvaluatorVariant PointEvaluatorFactory::CreatePointEvaluatorImpl(
  Type, const EvaluationRequest&) const
{
  throw pcms_error("CreatePointEvaluator: this source implements neither "
                   "BindImpl nor CreatePointEvaluatorImpl");
}

} // namespace pcms
