#include "pcms/field/function_space.h"

namespace pcms
{

const PointEvaluatorFactory& FunctionSpace::BindingSource() const noexcept
{
  if (evaluator_ == nullptr) {
    return *this;
  }
  return detail::BindingSourceOf(*evaluator_);
}

bool FunctionSpace::Supports(DifferentialOperator op) const noexcept
{
  return evaluator_ != nullptr && evaluator_->Supports(op);
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
    return evaluator_->CreatePointEvaluator<T>(std::move(bound), op);
  });
}

} // namespace pcms
