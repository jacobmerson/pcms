#ifndef POINT_CLOUD_LAYOUT_H_
#define POINT_CLOUD_LAYOUT_H_

#include "pcms/field.h"

namespace pcms
{

class PointCloudLayout : public FieldLayout
{
public:
  PointCloudLayout(int dim, Kokkos::View<Real**> coords,
                   CoordinateSystem coordinate_system);

  std::unique_ptr<FieldT<Real>> CreateField() const override;

  int GetNumComponents() const override;
  // nodes for standard lagrange FEM
  LO GetNumOwnedDofHolder() const override;
  GO GetNumGlobalDofHolder() const override;

  Rank1View<const bool, HostMemorySpace> GetOwned() const override;
  GlobalIDView<HostMemorySpace> GetGids() const override;
  CoordinateView<HostMemorySpace> GetDOFHolderCoordinates() const override;

  bool IsDistributed() override;
  size_t GetNumEnts() const;
  EntOffsetsArray GetEntOffsets() const override;

  FieldLayoutPlan BuildClientPlan(const redev::Partition& partition) const override;

  FieldLayoutPlan BuildServerPlan(
    GlobalIDView<HostMemorySpace> received_gids,
    const redev::InMessageLayout& incoming_layout, int mpi_rank,
    int mpi_size) const override;

  std::array<int, 4> GetNodesPerDim() const;

private:
  int dim_;
  int components_;
  CoordinateSystem coordinate_system_;
  Kokkos::View<Real**> coords_;
  Kokkos::View<bool*> owned_;
  Kokkos::View<GO*> gids_;
  Kokkos::View<bool*, HostMemorySpace> owned_host_;
  Kokkos::View<GO*, HostMemorySpace> gids_host_;
};
} // namespace pcms

#endif // POINT_CLOUD_LAYOUT_H_
