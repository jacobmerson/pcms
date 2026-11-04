#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H

// Backend types shared between MeshFieldsAdapter2 and
// MeshFieldsEvaluatorFactory. Extracted here to avoid a circular include.

#include <Kokkos_Core.hpp>
#include <MeshField.hpp>
#include <MeshField_Config.hpp>
#include <Omega_h_shape.hpp>
#include <memory>

#include "pcms/backends/omega_h_meshfields/layout.hpp"
#include "pcms/utility/types.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/field/field.hpp"

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
  /// @brief Gradient of the field at the points of evaluate.
  /// @param localCoords local coordinates of the points, grouped by element
  /// @param offsets offsets into `localCoords` of each element's points
  /// @return one row per point; column j * num_components + c holds d/dx_j of
  ///         component c
  virtual Kokkos::View<T**> evaluateGradient(
    Kokkos::View<T**> localCoords, Kokkos::View<LO*> offsets) const = 0;
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

  // grad f = J^{-1} df/dxi, with J_ij = dx_j/dxi_i from the (linear)
  // coordinate field, so the gradient is that of the field on the mesh's
  // straight-sided elements.
  Kokkos::View<T**> evaluateGradient(Kokkos::View<T**> localCoords,
                                     Kokkos::View<LO*> offsets) const override
  {
    if constexpr (Dim != 2) {
      throw pcms_error(
        "MeshFieldBackend: gradients are only supported on triangles");
    } else {
      auto self =
        const_cast<MeshFieldBackendImpl<T, Dim, Order, NumComponents>*>(this);
      const auto field_element =
        self->mesh_field_.CreateLagrangeElement(shape_field_.field);
      const auto coord_element = self->mesh_field_.CreateLagrangeElement(
        self->mesh_field_.getCoordField().field);
      Kokkos::View<T**> gradient("mesh_fields_gradient", localCoords.extent(0),
                                 Dim * NumComponents);
      Kokkos::parallel_for(
        "mesh_fields_gradient", static_cast<LO>(field_element.numMeshEnts),
        KOKKOS_LAMBDA(const LO ent) {
          const auto coords = coord_element.getNodeValues(ent);
          for (LO pt = offsets(ent); pt < offsets(ent + 1); ++pt) {
            Kokkos::Array<Real, Dim> xi;
            for (int d = 0; d < Dim; ++d) {
              xi[d] = localCoords(pt, d);
            }
            const auto coord_grads =
              coord_element.shapeFn.getLocalGradients(xi);
            Real jac[Dim][Dim] = {};
            for (size_t n = 0; n < coord_element.shapeFn.numNodes; ++n) {
              for (int i = 0; i < Dim; ++i) {
                for (int j = 0; j < Dim; ++j) {
                  jac[i][j] += coord_grads[n * Dim + i] * coords[n * Dim + j];
                }
              }
            }
            const Real det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
            const Real inv[Dim][Dim] = {{jac[1][1] / det, -jac[0][1] / det},
                                        {-jac[1][0] / det, jac[0][0] / det}};
            const auto grads = field_element.shapeFn.getLocalGradients(xi);
            for (size_t c = 0; c < NumComponents; ++c) {
              Real dfdxi[Dim] = {};
              for (auto topo : field_element.elm2dof.getTopology()) {
                for (size_t n = 0; n < field_element.shapeFn.numNodes; ++n) {
                  const auto map = field_element.elm2dof(n, c, ent, topo);
                  const Real value = field_element.field(
                    map.entity, map.node, map.component, map.topo);
                  for (int i = 0; i < Dim; ++i) {
                    dfdxi[i] += grads[n * Dim + i] * value;
                  }
                }
              }
              for (int j = 0; j < Dim; ++j) {
                gradient(pt, j * NumComponents + c) =
                  inv[j][0] * dfdxi[0] + inv[j][1] * dfdxi[1];
              }
            }
          }
        });
      return gradient;
    }
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

// ---------------------------------------------------------------------------
// Helper functors used in MeshFieldsBinding
// ---------------------------------------------------------------------------

struct ComputeOffsetsDeviceFunctor
{
  Kokkos::View<LO*> offsets_;
  Kokkos::View<LO*> elem_counts_;

  ComputeOffsetsDeviceFunctor(Kokkos::View<LO*> offsets,
                              Kokkos::View<LO*> elem_counts)
    : offsets_(offsets), elem_counts_(elem_counts)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(LO i, LO& partial, bool is_final) const
  {
    if (is_final) {
      offsets_(i) = partial;
    }
    partial += elem_counts_(i);
  }
};

struct CountPerElementFunctor
{
  Kokkos::View<const LO*> valid_indices_;
  Kokkos::View<LO*> owning_elem_ids_;
  Kokkos::View<LO*> elem_counts_;

  CountPerElementFunctor(Kokkos::View<const LO*> valid_indices,
                         Kokkos::View<LO*> owning_elem_ids,
                         Kokkos::View<LO*> elem_counts)
    : valid_indices_(valid_indices),
      owning_elem_ids_(owning_elem_ids),
      elem_counts_(elem_counts)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(LO i) const
  {
    LO orig_idx = valid_indices_(i);
    const auto owner_idx = owning_elem_ids_(orig_idx);
    Kokkos::atomic_add(&elem_counts_(owner_idx), 1);
  }
};

struct FillCoordinatesDeviceFunctor
{
  Kokkos::View<const LO*> valid_indices_;
  Kokkos::View<LO*> owning_elem_ids_;
  Kokkos::View<LO*> elem_counts_;
  Kokkos::View<LO*> offsets_;
  Kokkos::View<Real**> coordinates_;
  Kokkos::View<LO*> indices_;
  Omega_h::LOs tris2verts_;
  Omega_h::Reals mesh_coords_;
  Kokkos::View<const Real**, DeviceMemorySpace> global_coords_;
  Omega_h::Int dim_;

  FillCoordinatesDeviceFunctor(
    Kokkos::View<const LO*> valid_indices, Kokkos::View<LO*> owning_elem_ids,
    Kokkos::View<LO*> elem_counts, Kokkos::View<LO*> offsets,
    Kokkos::View<Real**> coordinates, Kokkos::View<LO*> indices,
    Omega_h::LOs tris2verts, Omega_h::Reals mesh_coords,
    Kokkos::View<const Real**, DeviceMemorySpace> global_coords,
    Omega_h::Int dim)
    : valid_indices_(valid_indices),
      owning_elem_ids_(owning_elem_ids),
      elem_counts_(elem_counts),
      offsets_(offsets),
      coordinates_(coordinates),
      indices_(indices),
      tris2verts_(tris2verts),
      mesh_coords_(mesh_coords),
      global_coords_(global_coords),
      dim_(dim)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(LO i) const
  {
    LO orig_idx = valid_indices_(i);
    const auto owner_idx = owning_elem_ids_(orig_idx);
    LO count =
      Kokkos::atomic_fetch_add(&elem_counts_(owner_idx), static_cast<LO>(-1));
    LO index = offsets_(owner_idx) + count - 1;
    const auto elem_tri2verts =
      Omega_h::gather_verts<3>(tris2verts_, owner_idx);
    const auto vertex_coords =
      Omega_h::gather_vectors<3, 2>(mesh_coords_, elem_tri2verts);
    const Omega_h::Vector<2> point{global_coords_(orig_idx, 0),
                                   global_coords_(orig_idx, 1)};
    const auto local =
      Omega_h::barycentric_from_global<2, 2>(point, vertex_coords);
    for (int j = 0; j < dim_; ++j) {
      coordinates_(index, j) = local[j];
    }
    indices_(index) = orig_idx;
  }
};

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_BACKEND_H