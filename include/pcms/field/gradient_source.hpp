#ifndef PCMS_FIELD_GRADIENT_SOURCE_HPP
#define PCMS_FIELD_GRADIENT_SOURCE_HPP

#include "pcms/field/point_evaluator_factory.hpp"
#include <memory>

namespace pcms
{

/// Presents the first derivatives of a source as a source: its evaluators
/// write the gradient of the field they are given, in FirstDerivativeBasis of
/// the field's stored basis. It binds through the wrapped source, so one bound
/// source serves evaluators of both.
class GradientSource final : public PointEvaluatorFactory
{
public:
  /// @param source source whose fields are differentiated; co-owned
  explicit GradientSource(std::shared_ptr<const PointEvaluatorFactory> source);

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
  std::shared_ptr<BoundSource> BindImpl(
    const EvaluationRequest& request) const override;
  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    int derivative_order) const override;

private:
  std::shared_ptr<const PointEvaluatorFactory> source_;
};

/// Source whose evaluators write the gradient of the field they are given;
/// see GradientSource.
[[nodiscard]] std::shared_ptr<const PointEvaluatorFactory> Gradient(
  std::shared_ptr<const PointEvaluatorFactory> source);

} // namespace pcms

#endif // PCMS_FIELD_GRADIENT_SOURCE_HPP
