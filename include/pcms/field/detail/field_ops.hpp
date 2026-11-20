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

/// field += addend, over the owned values ([owned][components]); leaves the
/// field consistent. (A free function: CUDA does not allow device lambdas in
/// the virtual functions of the evaluators that call it.)
template <typename T>
void AddToField(Field<T>& field,
                const Kokkos::View<T**, DeviceMemorySpace>& addend)
{
  const auto current = field.GetDOFHolderData().GetValues();
  Kokkos::View<T**, DeviceMemorySpace> sum(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "pcms::AddToField"),
    addend.extent(0), addend.extent(1));
  const auto n = static_cast<int>(sum.extent(0));
  const auto nc = static_cast<int>(sum.extent(1));
  Kokkos::parallel_for(
    "pcms::AddToField",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n * nc),
    KOKKOS_LAMBDA(const int k) {
      sum(k / nc, k % nc) = current(k / nc, k % nc) + addend(k / nc, k % nc);
    });
  field.SetDOFHolderDataUnchecked(MakeConstRank2View(sum));
  field.Synchronize();
}

} // namespace pcms::detail

#endif // PCMS_FIELD_DETAIL_FIELD_OPS_HPP
