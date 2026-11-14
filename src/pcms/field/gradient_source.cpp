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
  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    int derivative_order) const override
  {
    return apply_to_type(value_type, [&](auto type) -> PointEvaluatorVariant {
      return source_->CreatePointEvaluator<typename decltype(type)::type>(
        std::move(bound), derivative_order + 1);
    });
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
