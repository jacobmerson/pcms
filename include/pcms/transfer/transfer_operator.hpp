#ifndef PCMS_TRANSFER_TRANSFER_OPERATOR_HPP
#define PCMS_TRANSFER_TRANSFER_OPERATOR_HPP

#include "pcms/field/field.hpp"
#include "pcms/field/field_metadata.hpp"
#include "pcms/utility/assert.hpp"

namespace pcms
{

template <typename T>
class TransferOperator
{
public:
  using value_type = T;

  /// Writes `source` transferred to `target`. PCMS does not convert units:
  /// @throws pcms_error if both fields have units and they differ (see
  ///         CheckSameUnits); fields without units are not checked
  void Apply(const Field<T>& source, Field<T>& target) const
  {
    CheckSameUnits(source.GetMetadata(), target.GetMetadata(), "transfer");
    ApplyImpl(source, target);
  }

  /// @return whether ApplyTranspose is implemented
  [[nodiscard]] virtual bool SupportsTranspose() const noexcept
  {
    return false;
  }

  /// target = A^T source, where A is Apply: the adjoint of the linear map on
  /// the DOF vectors, <A x, y> = <x, A^T y> (coefficient dot products). Its
  /// source is data on Apply's target layout (loads, sensitivities); its
  /// target is a functional on Apply's source layout, overwritten (unlike
  /// PointEvaluator::EvaluateTranspose, which accumulates). The units of
  /// `source` and `target` are checked as for Apply.
  /// @throws pcms_error unless SupportsTranspose(), or if the units differ
  void ApplyTranspose(const Field<T>& source, Field<T>& target) const
  {
    CheckSameUnits(source.GetMetadata(), target.GetMetadata(),
                   "transposed transfer");
    ApplyTransposeImpl(source, target);
  }

  virtual ~TransferOperator() noexcept = default;

private:
  /// The transfer itself, after Apply's checks.
  virtual void ApplyImpl(const Field<T>& source, Field<T>& target) const = 0;

  virtual void ApplyTransposeImpl(const Field<T>& /*source*/,
                                  Field<T>& /*target*/) const
  {
    throw pcms_error("TransferOperator::ApplyTranspose: not supported by this "
                     "operator");
  }
};

} // namespace pcms

#endif // PCMS_TRANSFER_TRANSFER_OPERATOR_HPP
