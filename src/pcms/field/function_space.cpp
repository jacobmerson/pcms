#include "pcms/field/function_space.h"

namespace pcms
{

const PointEvaluatorFactory& FunctionSpace::BindingSource() const noexcept
{
  if (evaluator_ == nullptr) {
    return *this;
  }
  return evaluator_->BindingSource();
}

bool FunctionSpace::SupportsNearestBoundary() const noexcept
{
  return evaluator_ != nullptr && evaluator_->SupportsNearestBoundary();
}

std::shared_ptr<BoundSource> FunctionSpace::BindImpl(
  const EvaluationRequest&) const
{
  throw pcms_error("Bind: this function space does not evaluate fields at "
                   "points");
}

PointEvaluatorVariant FunctionSpace::CreatePointEvaluatorImpl(
  Type value_type, std::shared_ptr<const BoundSource> bound,
  int derivative_order) const
{
  if (evaluator_ == nullptr) {
    throw pcms_error("CreatePointEvaluator: this function space does not "
                     "evaluate fields at points");
  }
  return apply_to_type(value_type, [&](auto type) -> PointEvaluatorVariant {
    using T = typename decltype(type)::type;
    return evaluator_->CreatePointEvaluator<T>(std::move(bound),
                                               derivative_order);
  });
}

} // namespace pcms
