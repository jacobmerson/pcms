#ifndef PCMS_BACKENDS_XGC_LAYOUT_HPP
#define PCMS_BACKENDS_XGC_LAYOUT_HPP

#include "pcms/field/field_layout.hpp"
#include "pcms/backends/xgc/reverse_classification.hpp"

namespace pcms::xgc
{

class XGCFieldLayout : public FieldLayout
{
public:
  XGCFieldLayout(const ReverseClassificationVertex& reverse_classification,
                 LO num_plane_nodes, std::string name = "");

  int GetNumComponents() const override;
  LO GetNumDofHolder() const override;
  GO GetNumGlobalDofHolder() const override;
  Rank1View<const bool, DeviceMemorySpace> GetOwned() const override;
  GlobalIDView<HostMemorySpace> GetGidsHost() const override;
  bool IsDistributed() const override;
  EntOffsetsArray GetEntOffsets() const override;
  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates() const override;
  int GetDimension() const override;
  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationDimensions()
    const override;
  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationIds()
    const override;

  LO GetFullDataSize() const noexcept;

private:
  Kokkos::View<bool*, DeviceMemorySpace> owned_;
  Kokkos::View<GO*, DeviceMemorySpace> gids_;
  Kokkos::View<LO*, DeviceMemorySpace> class_dims_;
  Kokkos::View<LO*, DeviceMemorySpace> class_ids_;
  Kokkos::View<GO*, HostMemorySpace> gids_host_;
  Kokkos::View<Real**, DeviceMemorySpace> coords_;
  LO num_plane_nodes_;
};

} // namespace pcms::xgc

#endif // PCMS_BACKENDS_XGC_LAYOUT_HPP
