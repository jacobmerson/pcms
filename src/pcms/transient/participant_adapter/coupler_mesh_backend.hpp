#pragma once

#include "pcms/coupler/coupler.hpp"
#include "pcms/transient/coupling.hpp"

#include <Omega_h_mesh.hpp>

#include <any>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace pcms::transient
{

using CouplerInterfaceExtractor = std::function<InterfaceState(
  const Omega_h::Mesh&, const Field<Real>&)>;
using CouplerRegionPredicate =
  std::function<bool(std::span<const Real>)>;

struct CouplerMeshEndpoint
{
  Participant* participant;
  std::unique_ptr<Omega_h::Mesh> mesh;
  FieldHandle<Real> field;
  CouplerInterfaceExtractor extract_target_interface;
  CouplerRegionPredicate is_overlap;
};

// PCMS-backed coupling data plane. Participant fields arrive through
// FieldHandle::Receive, transfers run through Coupler::AddTransfer, and target
// fields leave through FieldHandle::Send. Only the compact boundary trace used
// by the transient fixed-point algorithm is extracted locally.
class CouplerMeshBackend final : public CouplingBackend
{
public:
  CouplerMeshBackend(CouplerMeshEndpoint left,
                     CouplerMeshEndpoint right,
                     TransferHandle left_to_right,
                     TransferHandle right_to_left);

  [[nodiscard]] InterfaceState Transfer(
    Participant& source, std::string_view source_interface,
    Participant& target, std::string_view target_interface,
    const InterfaceTransfer& transfer) override;

  [[nodiscard]] std::any Save() const override;
  void Restore(const std::any& checkpoint) override;
  void BeginWindow(Real time, Real dt) override;
  void BeginSweep(int sweep) override;
  void EndSweep(int sweep) override;
  void CommitWindow() override;
  void RejectWindow() override;
  [[nodiscard]] Real ResidualNorm(
    std::span<const Real> residual) const override;

  [[nodiscard]] std::size_t MeshCount() const noexcept;
  [[nodiscard]] std::size_t LeftOverlapDofCount() const noexcept;
  [[nodiscard]] std::size_t RightOverlapDofCount() const noexcept;
  [[nodiscard]] int CompletedSweeps() const noexcept;

private:
  struct SavedFields
  {
    std::vector<Real> left;
    std::vector<Real> right;
  };

  static CouplerMeshEndpoint ValidateEndpoint(
    CouplerMeshEndpoint endpoint);
  static std::vector<Real> CopyField(const Field<Real>& field);
  static void RestoreField(Field<Real>& field,
                           const std::vector<Real>& values);
  static std::size_t CountOverlap(
    const CouplerMeshEndpoint& endpoint);

  CouplerMeshEndpoint left_;
  CouplerMeshEndpoint right_;
  TransferHandle left_to_right_;
  TransferHandle right_to_left_;
  std::size_t left_overlap_dofs_;
  std::size_t right_overlap_dofs_;
  int active_sweep_ = -1;
  int completed_sweeps_ = 0;
  bool window_active_ = false;
};

} // namespace pcms::transient
