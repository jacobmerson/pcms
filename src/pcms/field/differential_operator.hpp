#ifndef PCMS_FIELD_DIFFERENTIAL_OPERATOR_HPP
#define PCMS_FIELD_DIFFERENTIAL_OPERATOR_HPP

#include <string_view>

namespace pcms
{

/// @brief The quantity a point evaluator computes from a field.
enum class DifferentialOperator
{
  Value,      ///< the field's values
  Gradient,   ///< first partial derivatives: one covariant index added
  Divergence, ///< contraction of the gradient: one index removed
  Curl,       ///< rank preserved; odd parity
  Hessian     ///< second partial derivatives: two covariant indices added
};

/// @return number of indices the operator adds to the stored basis
///         (negative removes)
[[nodiscard]] constexpr int RankChange(DifferentialOperator op) noexcept
{
  switch (op) {
    case DifferentialOperator::Value: return 0;
    case DifferentialOperator::Gradient: return 1;
    case DifferentialOperator::Divergence: return -1;
    case DifferentialOperator::Curl: return 0;
    case DifferentialOperator::Hessian: return 2;
  }
  return 0;
}

[[nodiscard]] constexpr std::string_view ToString(
  DifferentialOperator op) noexcept
{
  switch (op) {
    case DifferentialOperator::Value: return "Value";
    case DifferentialOperator::Gradient: return "Gradient";
    case DifferentialOperator::Divergence: return "Divergence";
    case DifferentialOperator::Curl: return "Curl";
    case DifferentialOperator::Hessian: return "Hessian";
  }
  return "?";
}

} // namespace pcms

#endif // PCMS_FIELD_DIFFERENTIAL_OPERATOR_HPP
