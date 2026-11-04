#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/function_space.hpp"
#include <string>

namespace pcms
{

std::shared_ptr<const BoundSource> PointEvaluatorFactory::Bind(
  CoordinateView<DeviceMemorySpace> coords, OutOfBoundsPolicy policy) const
{
  return BindThrough(BindRequest{coords, nullptr, policy});
}

std::shared_ptr<const BoundSource> PointEvaluatorFactory::Bind(
  const FunctionSpace& target, OutOfBoundsPolicy policy) const
{
  std::shared_ptr<const FieldLayout> layout = target.GetLayout();
  if (layout == nullptr) {
    throw pcms_error("Bind: the target space has no layout");
  }
  return BindThrough(
    BindRequest{layout->GetDOFHolderCoordinates(), layout, policy});
}

std::shared_ptr<const BoundSource> PointEvaluatorFactory::BindThrough(
  const BindRequest& request) const
{
  const PointEvaluatorFactory& binder = BindingSource();
  if (&binder != this) {
    return binder.BindThrough(request);
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

std::shared_ptr<BoundSource> PointEvaluatorFactory::BindImpl(
  const BindRequest&) const
{
  throw pcms_error("Bind: this source does not bind query points");
}

const PointEvaluatorFactory& detail::BindingSourceOf(
  const PointEvaluatorFactory& source) noexcept
{
  return source.BindingSource();
}

void PointEvaluatorFactory::CheckBoundByThis(const BoundSource& bound) const
{
  if (bound.bound_by_ != &BindingSource()) {
    throw pcms_error("CreatePointEvaluator: the bound source was bound by "
                     "another factory; bind this factory to the points");
  }
}

} // namespace pcms
