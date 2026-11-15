#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H

// Backend types shared between MeshFieldsAdapter2 and
// MeshFieldsEvaluatorFactory. Extracted here to avoid a circular include.

#include <Kokkos_Core.hpp>
#include <MeshField.hpp>
#include <MeshField_Config.hpp>
#include <Omega_h_shape.hpp>
#include <memory>

#include "pcms/field/layout/mesh_fields.h"
#include "pcms/utility/types.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"
#include "pcms/field/field.h"

namespace pcms
{

// ---------------------------------------------------------------------------
// Abstract backend interface
// ---------------------------------------------------------------------------
template <typename T>
class MeshFieldBackend
{
public:
  virtual ~MeshFieldBackend() = default;
  virtual Kokkos::View<T**> evaluate(Kokkos::View<T**> localCoords,
                                     Kokkos::View<LO*> offsets) const = 0;
  virtual void SetData(Rank1View<const T, DeviceMemorySpace> data,
                       size_t num_nodes, size_t num_components, int dim) = 0;
  virtual void GetData(Rank1View<T, DeviceMemorySpace> data, size_t num_nodes,
                       size_t num_components, int dim) const = 0;
};

// ---------------------------------------------------------------------------
// Concrete backend implementation
// ---------------------------------------------------------------------------
template <typename T, int Dim, int Order, int NumComponents>
class MeshFieldBackendImpl : public MeshFieldBackend<T>
{
public:
  MeshFieldBackendImpl(Omega_h::Mesh& mesh)
    : mesh_(mesh),
      mesh_field_(mesh),
      shape_field_(
        mesh_field_.template CreateLagrangeField<T, Order, NumComponents>())
  {
  }

  Kokkos::View<T**> evaluate(Kokkos::View<T**> localCoords,
                             Kokkos::View<LO*> offsets) const override
  {
    auto self =
      const_cast<MeshFieldBackendImpl<T, Dim, Order, NumComponents>*>(this);
    return self->mesh_field_.triangleLocalPointEval(localCoords, offsets,
                                                    shape_field_.field);
  }

  void SetData(Rank1View<const T, DeviceMemorySpace> data, size_t num_nodes,
               size_t num_components, int dim) override
  {
    size_t stride = num_nodes * num_components;
    auto topo = static_cast<MeshField::Mesh_Topology>(dim);
    Kokkos::parallel_for(
      mesh_.nents(dim), KOKKOS_CLASS_LAMBDA(size_t ent) {
        for (size_t n = 0; n < num_nodes; ++n) {
          for (size_t c = 0; c < num_components; ++c) {
            shape_field_.field(ent, n, c, topo) =
              data[ent * stride + n * num_components + c];
          }
        }
      });
  }

  void GetData(Rank1View<T, DeviceMemorySpace> data, size_t num_nodes,
               size_t num_components, int dim) const override
  {
    size_t stride = num_nodes * num_components;
    auto topo = static_cast<MeshField::Mesh_Topology>(dim);
    Kokkos::parallel_for(
      mesh_.nents(dim), KOKKOS_CLASS_LAMBDA(size_t ent) {
        for (size_t n = 0; n < num_nodes; ++n) {
          for (size_t c = 0; c < num_components; ++c) {
            data[ent * stride + n * num_components + c] =
              shape_field_.field(ent, n, c, topo);
          }
        }
      });
  }

private:
  Omega_h::Mesh& mesh_;
  MeshField::OmegahMeshField<DefaultExecutionSpace, Dim> mesh_field_;
  using FWC =
    decltype(mesh_field_
               .template CreateLagrangeField<T, Order, NumComponents>());
  FWC shape_field_; // FWC = FieldWithController; keeps ctrlr alive
};

// ---------------------------------------------------------------------------
// Factory function: create a MeshFieldBackend from a layout
// ---------------------------------------------------------------------------

// Helper to dispatch on num_components at runtime for a fixed (Dim, Order).
template <typename T, int Dim, int Order>
std::shared_ptr<MeshFieldBackend<T>> MakeBackendForComponents(
  Omega_h::Mesh& mesh, int num_components)
{
  switch (num_components) {
    case 1:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 1>>(mesh);
    case 2:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 2>>(mesh);
    case 3:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 3>>(mesh);
    case 4:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 4>>(mesh);
    case 5:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 5>>(mesh);
    case 6:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 6>>(mesh);
    case 7:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 7>>(mesh);
    case 8:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 8>>(mesh);
    case 9:
      return std::make_shared<MeshFieldBackendImpl<T, Dim, Order, 9>>(mesh);
    default:
      throw pcms_error("MeshFieldBackend: num_components " +
                       std::to_string(num_components) +
                       " exceeds maximum supported (9).");
  }
}

template <typename T>
std::shared_ptr<MeshFieldBackend<T>> MakeMeshFieldBackend(
  const MeshFieldsAdapterLayout& layout)
{
  if constexpr (!std::is_same_v<T, MeshField::Real4> &&
                !std::is_same_v<T, MeshField::Real8>) {
    throw pcms_error(
      "MeshFieldBackend only supports the MeshFields scalar types enabled in "
      "this build");
  }

  Omega_h::Mesh& mesh = layout.GetMesh();
  if (mesh.dim() == 3) {
    throw pcms_error("MeshFieldBackend does not support 3D meshes");
  }
  auto nodes_per_dim = layout.GetNodesPerDim();
  int num_components = layout.GetNumComponents();
  if (nodes_per_dim[0] == 1 && nodes_per_dim[1] == 0 && nodes_per_dim[2] == 0 &&
      nodes_per_dim[3] == 0) {
    switch (mesh.dim()) {
      case 1: return MakeBackendForComponents<T, 1, 1>(mesh, num_components);
      case 2: return MakeBackendForComponents<T, 2, 1>(mesh, num_components);
      default: break;
    }
  } else if (nodes_per_dim[0] == 1 && nodes_per_dim[1] == 1 &&
             nodes_per_dim[2] == 0 && nodes_per_dim[3] == 0) {
    switch (mesh.dim()) {
      case 2: return MakeBackendForComponents<T, 2, 2>(mesh, num_components);
      case 3: return MakeBackendForComponents<T, 3, 2>(mesh, num_components);
      default: break;
    }
  }
  return nullptr;
}

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H
