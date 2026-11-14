#ifndef PCMS_TRANSFER_TRANSFER_H
#define PCMS_TRANSFER_TRANSFER_H

#include "pcms/field/field.h"
#include "pcms/field/field_metadata.hpp"

namespace pcms
{

template <typename T>
class TransferOperator
{
public:
  using value_type = T;

  /// Writes `source` transferred to `target`. Throws if both fields have
  /// units and they differ (see CheckSameUnits).
  void Apply(const Field<T>& source, Field<T>& target) const
  {
    CheckSameUnits(source.GetMetadata(), target.GetMetadata(), "transfer");
    ApplyImpl(source, target);
  }

  virtual ~TransferOperator() noexcept = default;

private:
  /// The transfer itself, after Apply's checks.
  virtual void ApplyImpl(const Field<T>& source, Field<T>& target) const = 0;
};

} // namespace pcms

#endif
