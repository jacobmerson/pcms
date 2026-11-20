#include "pcms/field/field_metadata.hpp"
#include "pcms/utility/assert.hpp"

namespace pcms
{

void CheckSameUnits(const FieldMetadata& source, const FieldMetadata& target,
                    const std::string& operation)
{
  if (!source.units || !target.units || *source.units == *target.units) {
    return;
  }
  const std::string what = operation + " from '" + source.name + "' [" +
                           source.units->ToString() + "] to '" + target.name +
                           "' [" + target.units->ToString() + "]: ";
  if (!source.units->SameDimensions(*target.units)) {
    throw pcms_error(what + "the units have different dimensions");
  }
  throw pcms_error(what + "the units differ in scale or offset; PCMS does not "
                          "convert units, so convert one of the fields");
}

} // namespace pcms
