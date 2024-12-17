#ifndef PCMS_CLIENT_CPP_H
#define PCMS_CLIENT_CPP_H

#include <variant>

namespace pcms
{
    //namespace detail {
// Note that we have a closed set of types that can be used in the C interface
    using FieldAdapterVariant =
    std::variant<std::monostate, pcms::XGCFieldAdapter<double>,
                pcms::XGCFieldAdapter<float>, pcms::XGCFieldAdapter<int>,
                pcms::XGCFieldAdapter<long>,
                pcms::DummyFieldAdapter
    //#ifdef PCMS_HAS_OMEGA_H
    //               ,
    //               pcms::OmegaHFieldAdapter<double>,
    //               pcms::OmegaHFieldAdapter<int>
    //#endif
                >;
    //}

} // namespace pcms

template <typename T>
void pcms_create_xgc_field_adapter_t(
  const char* name, MPI_Comm comm, void* data, int size,
  const pcms::ReverseClassificationVertex& reverse_classification,
  in_overlap_function in_overlap, pcms::FieldAdapterVariant& field_adapter)
{
  PCMS_ALWAYS_ASSERT((size >0) ? (data!=nullptr) : true);
  pcms::ScalarArrayView<T, pcms::HostMemorySpace> data_view(
    reinterpret_cast<T*>(data), size);
  field_adapter.emplace<pcms::XGCFieldAdapter<T>>(
    name, comm, data_view, reverse_classification, in_overlap);
}

#endif