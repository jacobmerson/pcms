#include "pcms/field/function_space.h"

namespace pcms
{

bool FunctionSpace::Supports(DifferentialOperator op) const noexcept
{
  return evaluator_ != nullptr
           ? evaluator_->Supports(op)
           : (geometry_ != nullptr && op == DifferentialOperator::Value);
}

PointEvaluatorVariant FunctionSpace::CreatePointEvaluatorImpl(
  Type value_type, std::shared_ptr<const BoundSource> bound,
  DifferentialOperator op) const
{
  if (evaluator_ == nullptr) {
    throw pcms_error("CreatePointEvaluator: this function space does not "
                     "evaluate fields at points");
  }
  return apply_to_type(value_type, [&](auto type) -> PointEvaluatorVariant {
    using T = typename decltype(type)::type;
    return CreateUntransformedPointEvaluator<T>(*evaluator_, std::move(bound),
                                                op);
  });
}

} // namespace pcms
