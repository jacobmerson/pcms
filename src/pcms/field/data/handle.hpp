#ifndef PCMS_FIELD_DATA_HANDLE_HPP
#define PCMS_FIELD_DATA_HANDLE_HPP

#include "pcms/field/field_data.h"
#include "pcms/field/value_view.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/memory_spaces.h"
#include <string>
#include <utility>

namespace pcms
{

/// Field data whose values are held by an external library rather than by
/// PCMS. In place of DOF-holder values it carries a backend handle that names
/// the quantity inside that library (a field id and options, or an object the
/// library returned), so the field holds zero DOF holders: the getters return
/// [0][num_components] views and the setters throw.
///
/// The field is still a valid source for evaluation and transfers. It cannot
/// be received into, and sending it through the coupler sends no values.
template <typename T, typename Handle>
class HandleFieldData final : public FieldData<T>
{
public:
  /// @param handle names the quantity inside the external library
  /// @param basis value basis of the quantity's components
  /// @param num_components components per value; must match the layout of
  ///        the field this data is wrapped into
  HandleFieldData(Handle handle, ValueBasis basis, int num_components)
    : handle_(std::move(handle)),
      basis_(std::move(basis)),
      num_components_(num_components)
  {
    if (num_components_ < 1) {
      throw pcms_error("HandleFieldData: num_components must be positive");
    }
  }

  [[nodiscard]] const Handle& GetHandle() const noexcept { return handle_; }

  FieldValueType GetValueType() const override
  {
    return ValueTypeOfRank(basis_.Rank());
  }
  const ValueBasis& GetValueBasis() const override { return basis_; }

  Rank2View<const T, HostMemorySpace> GetDOFHolderDataHost() const override
  {
    return Rank2View<const T, HostMemorySpace>(static_cast<const T*>(nullptr),
                                               0, num_components_);
  }
  void SetDOFHolderDataHost(Rank2View<const T, HostMemorySpace>) override
  {
    ThrowReadOnly();
  }

  Rank2View<const T, DeviceMemorySpace> GetDOFHolderData() const override
  {
    return Rank2View<const T, DeviceMemorySpace>(static_cast<const T*>(nullptr),
                                                 0, num_components_);
  }
  void SetDOFHolderData(Rank2View<const T, DeviceMemorySpace>) override
  {
    ThrowReadOnly();
  }

private:
  [[noreturn]] static void ThrowReadOnly()
  {
    throw pcms_error("HandleFieldData: values are held by an external library "
                     "and cannot be set through PCMS");
  }

  Handle handle_;
  ValueBasis basis_;
  int num_components_;
};

} // namespace pcms

#endif // PCMS_FIELD_DATA_HANDLE_HPP
