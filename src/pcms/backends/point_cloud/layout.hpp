#ifndef PCMS_BACKENDS_POINT_CLOUD_LAYOUT_HPP
#define PCMS_BACKENDS_POINT_CLOUD_LAYOUT_HPP

#include "pcms/field/field.hpp"

namespace pcms::detail
{

class PointCloudLayout : public FieldLayout
{
public:
  /// A bare point cloud: no geometry (GetGeometry() is null), its points are
  /// vertices.
  PointCloudLayout(int dim, Kokkos::View<Real**> coords,
                   std::shared_ptr<const CoordinateSystem> coordinate_system);

  int GetNumComponents() const override;
  // nodes for standard lagrange FEM
  LO GetNumDofHolder() const override;
  GO GetNumGlobalDofHolder() const override;

  Rank1View<const bool, DeviceMemorySpace> GetOwned() const override;
  GlobalIDView<HostMemorySpace> GetGidsHost() const override;
  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates() const override;

  [[nodiscard]] bool IsDistributed() const override;
  size_t GetNumEnts() const;
  EntOffsetsArray GetEntOffsets() const override;

  int GetDimension() const override;

  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationDimensions()
    const override;

  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationIds()
    const override;

  std::array<int, 4> GetNodesPerDim() const;

  Kokkos::View<const Real**, DeviceMemorySpace> GetCoordinates() const;

private:
  int dim_;
  int components_;
  Kokkos::View<Real**> coords_;
  Kokkos::View<bool*> owned_;
  Kokkos::View<GO*> gids_;
  Kokkos::View<GO*, HostMemorySpace> gids_host_;
  Kokkos::View<LO*, DeviceMemorySpace> classification_dims_;
  Kokkos::View<LO*, DeviceMemorySpace> classification_ids_;
};
} // namespace pcms::detail

#endif // PCMS_BACKENDS_POINT_CLOUD_LAYOUT_HPP
