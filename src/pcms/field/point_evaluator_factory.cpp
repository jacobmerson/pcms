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

} // namespace pcms
