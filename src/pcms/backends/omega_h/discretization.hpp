#ifndef PCMS_DISCRETIZATION_DISCRETIZATION_OMEGA_H_H
#define PCMS_DISCRETIZATION_DISCRETIZATION_OMEGA_H_H

#include "pcms/discretization/discretization.hpp"

#include <Omega_h_array.hpp>
#include <Omega_h_mesh.hpp>

#include <array>
#include <memory>
#include <string>

namespace pcms::backend::omega_h
{

class OmegaHDiscretization : public Discretization
{
public:
  /// @param mesh the mesh, classified
  /// @param global_id_name name of the vertex, edge, ... tag that holds the
  ///        global entity ids the layouts on this discretization use
  /// @param owned_mask per-vertex ownership override: a vertex the mask is 0
  ///        for is not owned, whatever the mesh says; empty means the mesh's
  ///        own ownership. Layouts whose DOF holders are the mesh vertices
  ///        apply it; layouts of other entities do not.
  /// @throws pcms_error if a non-empty `owned_mask` does not have one entry
  ///         per vertex
  explicit OmegaHDiscretization(Omega_h::Mesh& mesh,
                                std::string global_id_name = "global",
                                Omega_h::Read<Omega_h::I8> owned_mask = {});

  bool SameEntities(const Discretization& other) const noexcept override;

  int GetDimension() const override;

  LO GetNumEntities(int entity_dim) const override;

  Rank1View<const ClassificationDimension, DeviceMemorySpace>
  GetEntityClassificationDimensions(int entity_dim) const override;

  Rank1View<const ClassificationId, DeviceMemorySpace>
  GetEntityClassificationIds(int entity_dim) const override;

  // The Omega_h mesh this discretization is defined on.
  Omega_h::Mesh& GetMesh() const noexcept { return mesh_; }

  /// @return the name of the tag that holds the global entity ids
  const std::string& GetGlobalIdName() const noexcept
  {
    return global_id_name_;
  }

  /// @return the per-vertex ownership override, empty if the mesh's own
  ///         ownership applies
  Omega_h::Read<Omega_h::I8> GetOwnedMask() const noexcept
  {
    return owned_mask_;
  }

private:
  static constexpr int max_entity_dim_ = Region;

  Omega_h::Mesh& mesh_;
  std::string global_id_name_;
  Omega_h::Read<Omega_h::I8> owned_mask_;
  std::array<Omega_h::Read<Omega_h::I8>, max_entity_dim_ + 1>
    class_dims_by_dim_;
  std::array<Omega_h::Read<Omega_h::ClassId>, max_entity_dim_ + 1>
    class_ids_by_dim_;
};

namespace detail
{

/// @brief The discretization a layout of `mesh` holds.
/// @param mesh the layout's mesh
/// @param discretization discretization to share; asserted not null
/// @param caller name of the layout, for the message
/// @return `discretization`
/// @throws pcms_error if `discretization` is of another mesh
std::shared_ptr<const OmegaHDiscretization> CheckDiscretizationOf(
  Omega_h::Mesh& mesh,
  std::shared_ptr<const OmegaHDiscretization> discretization,
  const char* caller);

} // namespace detail

} // namespace pcms::backend::omega_h

#endif // PCMS_DISCRETIZATION_DISCRETIZATION_OMEGA_H_H
