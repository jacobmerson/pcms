#include "client.h"
#include "pcms.h"
#include "pcms/field_adapter_interface.h"
#include "pcms/field_adapter_factory.h"
#include "pcms/adapter/xgc/xgc_field_adapter.h"
#include "pcms/dummy_field_adapter.h"
#include "pcms/adapter/omega_h/omega_h_field.h"
#include "pcms/adapter/xgc/xgc_reverse_classification.h"
#include "pcms/assert.h"
#include <variant>
#include <redev_variant_tools.h>
#include <fstream>
#include <memory>
// C API implementation uses the new inheritance model
// Store adapters as IFieldAdapter pointers instead of variants

[[nodiscard]] PcmsClientHandle pcms_create_client(const char* name,
                                                  MPI_Comm comm)
{
  auto* coupler = new pcms::Coupler(name, comm, false, {});
  auto* app = coupler->AddApplication(name);
  PcmsClientHandle handle;
  handle.couplerPointer = reinterpret_cast<void*>(coupler);
  handle.appPointer = reinterpret_cast<void*>(app);
  return handle;
}
void pcms_destroy_client(PcmsClientHandle client)
{
  if (client.couplerPointer != nullptr)
    delete reinterpret_cast<pcms::Coupler*>(client.couplerPointer);
}
PcmsReverseClassificationHandle pcms_load_reverse_classification(
  const char* file, MPI_Comm comm)
{
  // std::filesystem::path filepath{file};
  auto* rc = new pcms::ReverseClassificationVertex{
    pcms::ReadReverseClassificationVertex(file, comm)};
  return {reinterpret_cast<void*>(rc)};
}
void pcms_destroy_reverse_classification(PcmsReverseClassificationHandle rc)
{
  if (rc.pointer != nullptr)
    delete reinterpret_cast<pcms::ReverseClassificationVertex*>(rc.pointer);
}
// Helper function to add field using the new inheritance model
pcms::CoupledField* AddFieldFromAdapter(const char* name, pcms::Application* app,
                                        pcms::IFieldAdapter* adapter, bool participates) {
  if (!adapter) return nullptr;
  
  // We need a way to add the field with polymorphic adapter.
  // For now, let's try dynamic cast to determine the concrete type and use the appropriate factory
  auto adapter_type = adapter->GetAdapterType();
  
  switch (adapter_type) {
    case pcms::FieldAdapterType::DUMMY_FLOAT: {
      auto* dummy_adapter = dynamic_cast<pcms::DummyFieldAdapter<float>*>(adapter);
      if (dummy_adapter) {
        return app->AddField(name, *dummy_adapter, participates);
      }
      break;
    }
    case pcms::FieldAdapterType::DUMMY_DOUBLE: {
      auto* dummy_adapter = dynamic_cast<pcms::DummyFieldAdapter<double>*>(adapter);
      if (dummy_adapter) {
        return app->AddField(name, *dummy_adapter, participates);
      }
      break;
    }
    case pcms::FieldAdapterType::XGC_FLOAT: {
      auto* xgc_adapter = dynamic_cast<pcms::XGCFieldAdapter<float>*>(adapter);
      if (xgc_adapter) {
        return app->AddField(name, *xgc_adapter, participates);
      }
      break;
    }
    case pcms::FieldAdapterType::XGC_DOUBLE: {
      auto* xgc_adapter = dynamic_cast<pcms::XGCFieldAdapter<double>*>(adapter);
      if (xgc_adapter) {
        return app->AddField(name, *xgc_adapter, participates);
      }
      break;
    }
    case pcms::FieldAdapterType::XGC_INT: {
      auto* xgc_adapter = dynamic_cast<pcms::XGCFieldAdapter<int>*>(adapter);
      if (xgc_adapter) {
        return app->AddField(name, *xgc_adapter, participates);
      }
      break;
    }
    case pcms::FieldAdapterType::XGC_LONG_INT: {
      auto* xgc_adapter = dynamic_cast<pcms::XGCFieldAdapter<long int>*>(adapter);
      if (xgc_adapter) {
        return app->AddField(name, *xgc_adapter, participates);
      }
      break;
    }
    case pcms::FieldAdapterType::OMEGA_H_FLOAT: {
      auto* omega_h_adapter = dynamic_cast<pcms::OmegaHFieldAdapter<float>*>(adapter);
      if (omega_h_adapter) {
        return app->AddField(name, *omega_h_adapter, participates);
      }
      break;
    }
    case pcms::FieldAdapterType::OMEGA_H_DOUBLE: {
      auto* omega_h_adapter = dynamic_cast<pcms::OmegaHFieldAdapter<double>*>(adapter);
      if (omega_h_adapter) {
        return app->AddField(name, *omega_h_adapter, participates);
      }
      break;
    }
    default:
      return nullptr;
  }
  
  return nullptr;
}

PcmsFieldHandle pcms_add_field(PcmsClientHandle client_handle, const char* name,
                               PcmsFieldAdapterHandle adapter_handle,
                               int participates)
{
  auto* adapter = reinterpret_cast<pcms::IFieldAdapter*>(adapter_handle.pointer);
  auto* app = reinterpret_cast<pcms::Application*>(client_handle.appPointer);
  PCMS_ALWAYS_ASSERT(app != nullptr);
  PCMS_ALWAYS_ASSERT(adapter != nullptr);
  
  pcms::CoupledField* field = AddFieldFromAdapter(name, app, adapter, participates);
  
  PcmsFieldHandle handle;
  handle.pointer = reinterpret_cast<void*>(field);
  handle.type = adapter_handle.type;
  return handle;
}
void pcms_send_field_name(PcmsClientHandle client_handle, const char* name)
{
  auto* app = reinterpret_cast<pcms::Application*>(client_handle.appPointer);
  PCMS_ALWAYS_ASSERT(app != nullptr);
  app->SendField(name);
}
void pcms_receive_field_name(PcmsClientHandle client_handle, const char* name)
{
  auto* app = reinterpret_cast<pcms::Application*>(client_handle.appPointer);
  PCMS_ALWAYS_ASSERT(app != nullptr);
  app->ReceiveField(name);
}
void pcms_send_field(PcmsFieldHandle field_handle)
{
  auto* field = reinterpret_cast<pcms::CoupledField*>(field_handle.pointer);
  PCMS_ALWAYS_ASSERT(field != nullptr);
  field->Send();
}
void pcms_receive_field(PcmsFieldHandle field_handle)
{
  auto* field = reinterpret_cast<pcms::CoupledField*>(field_handle.pointer);
  PCMS_ALWAYS_ASSERT(field != nullptr);
  field->Receive();
}
template <typename T>
std::unique_ptr<pcms::IFieldAdapter> pcms_create_xgc_field_adapter_t(
  const char* name, MPI_Comm comm, void* data, int size,
  const pcms::ReverseClassificationVertex& reverse_classification,
  in_overlap_function in_overlap)
{
  PCMS_ALWAYS_ASSERT((size > 0) ? (data != nullptr) : true);
  pcms::Rank1View<T, pcms::HostMemorySpace> data_view(
    reinterpret_cast<T*>(data), size);
  
  std::function<int8_t(int, int)> overlap_func = in_overlap;
  return pcms::CreateXGCFieldAdapter<T>(name, comm, data_view, reverse_classification, overlap_func);
}
PcmsFieldAdapterHandle pcms_create_xgc_field_adapter(
  const char* name, MPI_Comm comm, void* data, int size, PcmsType data_type,
  const PcmsReverseClassificationHandle rc, in_overlap_function in_overlap)
{
  PCMS_ALWAYS_ASSERT(rc.pointer != nullptr);
  auto* reverse_classification =
    reinterpret_cast<const pcms::ReverseClassificationVertex*>(rc.pointer);
  PCMS_ALWAYS_ASSERT(reverse_classification != nullptr);
  
  std::unique_ptr<pcms::IFieldAdapter> field_adapter;
  PcmsAdapterType adapter_type;
  
  switch (data_type) {
    case PCMS_DOUBLE:
      field_adapter = pcms_create_xgc_field_adapter_t<double>(
        name, comm, data, size, *reverse_classification, in_overlap);
      adapter_type = PCMS_ADAPTER_XGC;
      break;
    case PCMS_FLOAT:
      field_adapter = pcms_create_xgc_field_adapter_t<float>(
        name, comm, data, size, *reverse_classification, in_overlap);
      adapter_type = PCMS_ADAPTER_XGC;
      break;
    case PCMS_INT:
      field_adapter = pcms_create_xgc_field_adapter_t<int>(
        name, comm, data, size, *reverse_classification, in_overlap);
      adapter_type = PCMS_ADAPTER_XGC;
      break;
    case PCMS_LONG_INT:
      field_adapter = pcms_create_xgc_field_adapter_t<long int>(
        name, comm, data, size, *reverse_classification, in_overlap);
      adapter_type = PCMS_ADAPTER_XGC;
      break;
    default:
      PCMS_ALWAYS_ASSERT(false, MPI_COMM_WORLD, "trying to create XGC adapter with invalid type!\n");
  }
  
  PcmsFieldAdapterHandle handle;
  handle.pointer = reinterpret_cast<void*>(field_adapter.release()); // Transfer ownership
  handle.type = adapter_type;
  return handle;
}
PcmsFieldAdapterHandle pcms_create_dummy_field_adapter()
{
  auto field_adapter = pcms::CreateDummyDoubleAdapter("dummy");
  
  PcmsFieldAdapterHandle handle;
  handle.pointer = reinterpret_cast<void*>(field_adapter.release()); // Transfer ownership
  handle.type = PCMS_ADAPTER_GEM; // Use GEM as dummy type
  return handle;
}

void pcms_destroy_field_adapter(PcmsFieldAdapterHandle adapter_handle)
{
  auto* adapter = reinterpret_cast<pcms::IFieldAdapter*>(adapter_handle.pointer);
  if (adapter != nullptr) {
    delete adapter;
  }
}

// Additional validation functions
int pcms_is_valid_field_adapter(PcmsFieldAdapterHandle handle)
{
  return (handle.pointer != nullptr) ? 1 : 0;
}

int pcms_is_valid_field(PcmsFieldHandle handle)
{
  return (handle.pointer != nullptr) ? 1 : 0;
}

PcmsAdapterType pcms_get_field_adapter_type(PcmsFieldAdapterHandle handle)
{
  if (handle.pointer == nullptr) {
    return PCMS_ADAPTER_GEM; // Return dummy type for invalid handles
  }
  return handle.type;
}
int pcms_reverse_classification_count_verts(PcmsReverseClassificationHandle rc)
{
  auto* reverse_classification =
    reinterpret_cast<const pcms::ReverseClassificationVertex*>(rc.pointer);
  PCMS_ALWAYS_ASSERT(reverse_classification != nullptr);
  return std::accumulate(reverse_classification->begin(),
                         reverse_classification->end(), 0,
                         [](auto current, const auto& verts) {
                           return current + verts.second.size();
                         });
}
void pcms_begin_send_phase(PcmsClientHandle h)
{
  auto* app = reinterpret_cast<pcms::Application*>(h.appPointer);
  PCMS_ALWAYS_ASSERT(app != nullptr);
  app->BeginSendPhase();
}
void pcms_end_send_phase(PcmsClientHandle h)
{
  auto* app = reinterpret_cast<pcms::Application*>(h.appPointer);
  PCMS_ALWAYS_ASSERT(app != nullptr);
  app->EndSendPhase();
}
void pcms_begin_receive_phase(PcmsClientHandle h)
{
  auto* app = reinterpret_cast<pcms::Application*>(h.appPointer);
  PCMS_ALWAYS_ASSERT(app != nullptr);
  app->BeginReceivePhase();
}
void pcms_end_receive_phase(PcmsClientHandle h)
{
  auto* app = reinterpret_cast<pcms::Application*>(h.appPointer);
  PCMS_ALWAYS_ASSERT(app != nullptr);
  app->EndReceivePhase();
}
