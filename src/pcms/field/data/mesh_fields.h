#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_FIELD_DATA_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_FIELD_DATA_H

#include "pcms/field/layout/mesh_fields.h"
#include "pcms/field/evaluator/mesh_fields_backend.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_metadata.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"

#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

template <typename T>
class MeshFieldsFieldData : public FieldData<T>
{
public:
  MeshFieldsFieldData(std::shared_ptr<const MeshFieldsAdapterLayout> layout,
                      FieldMetadata metadata)
    : layout_(std::move(layout)),
      metadata_(metadata),
      mesh_field_(MakeMeshFieldBackend<T>(*layout_)),
      host_data_("meshfields_field_data",
                 static_cast<size_t>(layout_->GetNumOwnedDofHolder()),
                 static_cast<size_t>(layout_->GetNumComponents())),
      device_data_("meshfields_field_data_device",
                   static_cast<size_t>(layout_->GetNumOwnedDofHolder()),
                   static_cast<size_t>(layout_->GetNumComponents()))
  {
    if (!mesh_field_) {
      throw pcms_error(
        "MeshFieldsFieldData does not support this layout/order");
    }
  }

  const FieldMetadata& GetMetadata() const override { return metadata_; }

  Rank2View<const T, HostMemorySpace> GetDOFHolderDataHost() const override
  {
    DeepCopyMismatchLayouts(host_data_, device_data_);
    return MakeConstRank2View(host_data_);
  }

  void SetDOFHolderDataHost(Rank2View<const T, HostMemorySpace> values) override
  {
    CopyHostRank2ViewToDeviceView(device_data_, values);
    SyncBackend(GetDOFHolderData());
  }

  Rank2View<const T, DeviceMemorySpace> GetDOFHolderData() const override
  {
    return MakeConstRank2View(device_data_);
  }

  void SetDOFHolderData(Rank2View<const T, DeviceMemorySpace> values) override
  {
    CopyDeviceRank2ViewToDeviceView(device_data_, values);
    SyncBackend(GetDOFHolderData());
  }

  std::shared_ptr<MeshFieldBackend<T>> GetMeshFieldBackend() const
  {
    return mesh_field_;
  }

private:
  // Serialization boundary: meshfields' SetData consumes flat node-major
  // spans, so the shaped data is explicitly repacked into a flat node-major
  // staging buffer here — the one place this backend handles flat memory.
  void SyncBackend(Rank2View<const T, DeviceMemorySpace> data)
  {
    auto nodes_per_dim = layout_->GetNodesPerDim();
    auto num_components = layout_->GetNumComponents();
    auto& mesh = layout_->GetMesh();
    Kokkos::View<T*, DeviceMemorySpace> flat(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "meshfields_sync_staging"),
      static_cast<size_t>(data.extent(0)) * data.extent(1));
    CopyDeviceRank2ViewToDeviceView(flat, data);
    // Each mesh dimension owns a contiguous block of node-major rows.
    size_t row_offset = 0;
    for (int i = 0; i <= mesh.dim(); ++i) {
      if (nodes_per_dim[i]) {
        size_t num_rows = static_cast<size_t>(mesh.nents(i)) *
                          static_cast<size_t>(nodes_per_dim[i]);
        size_t len = num_rows * static_cast<size_t>(num_components);
        Rank1View<const T, DeviceMemorySpace> subspan{
          flat.data() + row_offset * static_cast<size_t>(num_components), len};
        mesh_field_->SetData(subspan, nodes_per_dim[i], num_components, i);
        row_offset += num_rows;
      }
    }
  }

  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  FieldMetadata metadata_;
  std::shared_ptr<MeshFieldBackend<T>> mesh_field_;
  mutable Kokkos::View<T**, HostMemorySpace> host_data_;
  Kokkos::View<T**, DeviceMemorySpace> device_data_;
};

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_FIELD_DATA_H
