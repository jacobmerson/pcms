#ifndef PCMS_BACKENDS_MFEM_LAYOUT_HPP
#define PCMS_BACKENDS_MFEM_LAYOUT_HPP

#include "pcms/utility/coordinate_system.hpp"
#include "pcms/field/field_layout.hpp"

#include <mfem.hpp>

#include <memory>

namespace pcms::mfem
{

// Field layout for an MFEM order-1 H1 (vertex) scalar field.
class MFEMLayout : public FieldLayout
{
public:
  MFEMLayout(::mfem::ParMesh& pmesh, ::mfem::ParFiniteElementSpace& pfes,
             std::shared_ptr<const CoordinateSystem> coordinate_system,
             std::string name = "");

  int GetNumComponents() const override;
  LO GetNumDofHolder() const override;
  GO GetNumGlobalDofHolder() const override;

  Rank1View<const bool, DeviceMemorySpace> GetOwned() const override;
  GlobalIDView<HostMemorySpace> GetGidsHost() const override;
  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates() const override;

  [[nodiscard]] bool IsDistributed() const override;
  EntOffsetsArray GetEntOffsets() const override;
  int GetDimension() const override;

  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationDimensions()
    const override;

  Rank1View<const LO, DeviceMemorySpace> GetDOFHolderClassificationIds()
    const override;

  ::mfem::ParMesh& GetMesh() const noexcept { return pmesh_; }
  ::mfem::ParFiniteElementSpace& GetFESpace() const noexcept { return pfes_; }

  /// @brief A per-vertex overlap mask from MFEM element attributes
  static Kokkos::View<bool*, DeviceMemorySpace> OverlapMaskFromAttribute(
    ::mfem::ParMesh& pmesh, int attribute);

private:
  void AssertVertexScalarSpace() const;

  ::mfem::ParMesh& pmesh_;
  ::mfem::ParFiniteElementSpace& pfes_;
  int dim_;

  Kokkos::View<Real**, DeviceMemorySpace> coords_;
  Kokkos::View<bool*, DeviceMemorySpace> owned_;
  Kokkos::View<GO*, HostMemorySpace> gids_host_;
  Kokkos::View<LO*, DeviceMemorySpace> classification_dims_;
  Kokkos::View<LO*, DeviceMemorySpace> classification_ids_;
};

} // namespace pcms::mfem

#endif // PCMS_BACKENDS_MFEM_LAYOUT_HPP
