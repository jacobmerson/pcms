#ifndef PCMS_BACKENDS_OMEGA_H_MESHFIELDS_LAYOUT_HPP
#define PCMS_BACKENDS_OMEGA_H_MESHFIELDS_LAYOUT_HPP

#include <Omega_h_mesh.hpp>

#include "pcms/utility/arrays.hpp"
#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/field/field_layout.hpp"
#include "pcms/field/field.hpp"

#include <array>

namespace pcms::omega_h_meshfields
{
class MeshFieldsAdapterLayout : public FieldLayout
{
public:
  /// The layout is on `geometry`, whose mesh it lays out, and takes its
  /// coordinate system and its global id tag name; it does not apply an owned
  /// mask.
  /// @throws pcms_error if `geometry` is null
  /// @param name the layout's name (FieldLayout::GetName)
  MeshFieldsAdapterLayout(
    std::shared_ptr<const omega_h::OmegaHGeometry> geometry,
    std::array<int, 4> nodes_per_dim, int num_components,
    std::string name = "");

  int GetNumComponents() const override;
  // nodes for standard lagrange FEM
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

  // returns true if the field layout is distributed: on several ranks it may
  // hold ghost DOF holders in addition to the owned ones (see HasGhosts)
  [[nodiscard]] bool IsDistributed() const override;

  int GetDimension() const override;
  int GetDOFHolderEntityDim() const override;

  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationDimensions()
    const override;
  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationIds()
    const override;

  std::array<int, 4> GetNodesPerDim() const;
  size_t GetNumEnts() const;
  Omega_h::Mesh& GetMesh() const;

private:
  Omega_h::Mesh& mesh_;
  Omega_h::Write<Omega_h::GO> gids_;
  int num_components_;
  std::array<int, 4> nodes_per_dim_;
  Kokkos::View<Real**> dof_holder_coords_;
  Omega_h::Write<Omega_h::ClassId> class_ids_;
  Omega_h::Write<Omega_h::I8> class_dims_;
  Kokkos::View<bool*> owned_;
  LO num_owned_ = 0;
  Kokkos::View<GO*, HostMemorySpace> owned_gids_host_;
  Kokkos::View<Real**, DeviceMemorySpace> owned_coords_2d_;
  Kokkos::View<LO*, DeviceMemorySpace> owned_indices_;

  pcms::detail::OwnedClassification owned_classification_;

  // Entity offsets over the DOF holders with ghosts.
  EntOffsetsArray EntOffsetsWithGhosts() const;
  void BuildOwnedViews();
};

} // namespace pcms::omega_h_meshfields
#endif // PCMS_BACKENDS_OMEGA_H_MESHFIELDS_LAYOUT_HPP
