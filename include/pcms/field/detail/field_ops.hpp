#ifndef PCMS_FIELD_DETAIL_FIELD_OPS_HPP
#define PCMS_FIELD_DETAIL_FIELD_OPS_HPP

#include "pcms/field/field.hpp"
#include "pcms/utility/assert.hpp"
#include <Kokkos_Core.hpp>
#include <string>

namespace pcms::detail
{

template <typename T>
void CheckSameLayout(const Field<T>& a, const Field<T>& b, const char* what)
{
  if (&a.GetLayout() != &b.GetLayout()) {
    throw pcms_error(std::string(what) +
                     ": the fields must have the same layout");
  }
}

/// Sets every owned value of the field to zero.
template <typename T>
void SetZero(Field<T>& field)
{
  Kokkos::View<T**, DeviceMemorySpace> zeros(
    "pcms::SetZero", field.GetLayout().GetNumDofHolder(),
    field.GetLayout().GetNumComponents());
  field.SetDOFHolderDataUnchecked(MakeConstRank2View(zeros));
  field.Synchronize();
}

} // namespace pcms::detail

#endif // PCMS_FIELD_DETAIL_FIELD_OPS_HPP
