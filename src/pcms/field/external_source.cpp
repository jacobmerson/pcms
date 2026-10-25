#include "pcms/field/external_source.hpp"
#include "pcms/utility/assert.h"

namespace pcms
{

ExternalSource::ExternalSource(std::shared_ptr<const CoordinateSystem> system,
                               std::string name)
  : system_(std::move(system)), name_(std::move(name))
{
  if (system_ == nullptr) {
    throw pcms_error(name_ + ": coordinate system must not be null");
  }
  layout_ = std::make_shared<EmptyFieldLayout>(system_);
}

FieldVariant ExternalSource::CreateFieldImpl(Type /*storage_type*/,
                                             ValueBasis /*basis*/) const
{
  throw pcms_error(name_ +
                   ": a value type and basis do not name a quantity of this "
                   "source; use the source's own field constructor");
}

FieldVariant ExternalSource::CreateFieldImpl(FieldDataVariant /*data*/) const
{
  throw pcms_error(name_ +
                   ": fields of this source name quantities held by an "
                   "external library; use the source's own field constructor");
}

} // namespace pcms
