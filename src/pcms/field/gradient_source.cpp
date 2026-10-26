#include "pcms/field/gradient_source.hpp"

namespace pcms
{
namespace
{

/// @brief Source whose evaluators write the first derivatives of the wrapped
///        source's evaluators.
class GradientSource final : public PointEvaluatorFactory
{
public:
  /// @param source source to differentiate; not null
  explicit GradientSource(std::shared_ptr<const PointEvaluatorFactory> source)
    : source_(std::move(source))
  {
  }

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const override
  {
    return source_->GetCoordinateSystem();
  }

  [[nodiscard]] const PointEvaluatorFactory& BindingSource()
    const noexcept override
  {
    return source_->BindingSource();
  }

protected:
  std::shared_ptr<BoundSource> BindImpl(const EvaluationRequest&) const override
  {
    // Bind forwards to BindingSource(), the wrapped source.
    throw pcms_error("Gradient: binds through the wrapped source");
  }

  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    int derivative_order) const override
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
    throw pcms_error("Gradient: unsupported value type");
  }

private:
  std::shared_ptr<const PointEvaluatorFactory> source_;
};

} // namespace

std::shared_ptr<const PointEvaluatorFactory> Gradient(
  std::shared_ptr<const PointEvaluatorFactory> source)
{
  if (source == nullptr) {
    throw pcms_error("Gradient: source must not be null");
  }
  return std::make_shared<GradientSource>(std::move(source));
}

} // namespace pcms
