#ifndef PCMS_BACKENDS_OMEGA_H_ENTITY_LAYOUT_HPP
#define PCMS_BACKENDS_OMEGA_H_ENTITY_LAYOUT_HPP

#include <Omega_h_mesh.hpp>

#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/field/field_layout.hpp"

namespace pcms::omega_h
{

class OmegaHEntityLayout : public FieldLayout
{
public:
  /// The DOF holders are the entities of dimension `entity_dim` of the
  /// geometry's mesh. The layout is on `geometry`, and takes its coordinate
  /// system and its global id tag name.
  /// @throws pcms_error if `geometry` is null or not an Omega_h geometry
  OmegaHEntityLayout(std::shared_ptr<const Geometry> geometry, int entity_dim,
                     int num_components);

  int GetNumComponents() const override;
  LO GetNumDofHolder() const override;
  [[nodiscard]] bool HasGhosts() const override;
  // Number of DOF holders, owned and ghost: the size of the field data
  // this layout's backends store.
  LO GetNumDofHolderWithGhosts() const;
  GO GetNumGlobalDofHolder() const override;

  Rank1View<const bool, DeviceMemorySpace> GetOwned() const override;
  GlobalIDView<HostMemorySpace> GetGidsHost() const override;
  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates() const override;
  EntOffsetsArray GetEntOffsets() const override;
  Kokkos::View<const LO*, DeviceMemorySpace> GetOwnedIndices() const override;

  [[nodiscard]] bool IsDistributed() const override;
  int GetDimension() const override;
  int GetDOFHolderEntityDim() const override;

  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationDimensions()
    const override;
  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationIds()
    const override;

private:
  Omega_h::Mesh& mesh_;
  int dimension_;
  int entity_dim_;
  int num_components_;
  GO num_global_dof_holder_;

  Omega_h::Write<Omega_h::GO> gids_;
  Omega_h::Read<Real> coords_; // device coordinates (1D flattened)
  Kokkos::View<Real**, DeviceMemorySpace> coords_2d_; // device coordinates (2D)
  Omega_h::Read<Omega_h::ClassId> class_ids_;
  Omega_h::Read<Omega_h::I8> class_dims_;
  Kokkos::View<bool*, DeviceMemorySpace> owned_;
  LO num_owned_ = 0;
  Kokkos::View<GO*, HostMemorySpace> owned_gids_host_;
  Kokkos::View<Real**, DeviceMemorySpace> owned_coords_2d_;
  Kokkos::View<LO*, DeviceMemorySpace> owned_indices_;

  pcms::detail::OwnedClassification owned_classification_;

  // Entity offsets over the DOF holders with ghosts.
  EntOffsetsArray EntOffsetsWithGhosts() const;
  void BuildOwnedViews();
};

} // namespace pcms::omega_h

#endif // PCMS_BACKENDS_OMEGA_H_ENTITY_LAYOUT_HPP
