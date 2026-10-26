#include "pcms/field/gradient_source.hpp"

namespace pcms
{

GradientSource::GradientSource(
  std::shared_ptr<const PointEvaluatorFactory> source)
  : source_(std::move(source))
{
  if (source_ == nullptr) {
    throw pcms_error("GradientSource: source must not be null");
  }
}

std::shared_ptr<BoundSource> GradientSource::BindImpl(
  const EvaluationRequest&) const
{
  // Bind forwards to BindingSource(), the wrapped source.
  throw pcms_error("GradientSource: binds through the wrapped source");
}

PointEvaluatorVariant GradientSource::CreatePointEvaluatorImpl(
  Type value_type, std::shared_ptr<const BoundSource> bound,
  int derivative_order) const
{
  const int order = derivative_order + 1;
  switch (value_type) {
    case Type::Real:
      return source_->CreatePointEvaluator<Real>(std::move(bound), order);
    case Type::LO:
      return source_->CreatePointEvaluator<LO>(std::move(bound), order);
    case Type::GO:
      return source_->CreatePointEvaluator<GO>(std::move(bound), order);
    case Type::Int8:
      return source_->CreatePointEvaluator<int8_t>(std::move(bound), order);
    case Type::Float:
      return source_->CreatePointEvaluator<float>(std::move(bound), order);
  }
  throw pcms_error("GradientSource::CreatePointEvaluator: unsupported value "
                   "type");
}

std::shared_ptr<const PointEvaluatorFactory> Gradient(
  std::shared_ptr<const PointEvaluatorFactory> source)
{
  return std::make_shared<GradientSource>(std::move(source));
}

} // namespace pcms
