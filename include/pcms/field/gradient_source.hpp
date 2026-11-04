#ifndef PCMS_FIELD_GRADIENT_SOURCE_HPP
#define PCMS_FIELD_GRADIENT_SOURCE_HPP

#include "pcms/field/point_evaluator_factory.hpp"
#include <memory>

namespace pcms
{

/// @brief Wraps a source so that its evaluators write the gradient of a field.
/// @param source source to differentiate; must support derivative order 1
/// @return source whose evaluators write FirstDerivativeBasis of the field's
///         stored basis, and that binds through `source`
/// @throws pcms_error if `source` is null
[[nodiscard]] std::shared_ptr<const PointEvaluatorFactory> Gradient(
  std::shared_ptr<const PointEvaluatorFactory> source);

} // namespace pcms

#endif // PCMS_FIELD_GRADIENT_SOURCE_HPP
