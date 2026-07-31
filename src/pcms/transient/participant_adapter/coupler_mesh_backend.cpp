#include "pcms/transient/participant_adapter/coupler_mesh_backend.hpp"

#include "pcms/utility/arrays.h"

#include <Omega_h_array.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace pcms::transient
{

CouplerMeshBackend::CouplerMeshBackend(
  CouplerMeshEndpoint left, CouplerMeshEndpoint right,
  TransferHandle left_to_right, TransferHandle right_to_left)
  : left_(ValidateEndpoint(std::move(left))),
    right_(ValidateEndpoint(std::move(right))),
    left_to_right_(std::move(left_to_right)),
    right_to_left_(std::move(right_to_left)),
    left_overlap_dofs_(CountOverlap(left_)),
    right_overlap_dofs_(CountOverlap(right_))
{
  if (left_overlap_dofs_ == 0 || right_overlap_dofs_ == 0)
    throw std::invalid_argument(
      "CouplerMeshBackend: overlap is empty");
  if (left_overlap_dofs_ ==
        static_cast<std::size_t>(left_.mesh->nverts()) ||
      right_overlap_dofs_ ==
        static_cast<std::size_t>(right_.mesh->nverts())) {
    throw std::invalid_argument(
      "CouplerMeshBackend: overlap must not cover an entire coupler mesh");
  }
}

InterfaceState CouplerMeshBackend::Transfer(
  Participant& source, std::string_view source_interface,
  Participant& target, std::string_view,
  const InterfaceTransfer& transfer)
{
  if (&source == left_.participant && &target == right_.participant) {
    // RemoteParticipant::GetInterface coordinates a PCMS Field receive. No
    // field values are returned through InterfaceState.
    static_cast<void>(source.GetInterface(source_interface));
    left_to_right_.Run();
    return transfer(right_.extract_target_interface(
      *right_.mesh, right_.field.GetField()));
  }
  if (&source == right_.participant && &target == left_.participant) {
    static_cast<void>(source.GetInterface(source_interface));
    right_to_left_.Run();
    return transfer(left_.extract_target_interface(
      *left_.mesh, left_.field.GetField()));
  }
  throw std::invalid_argument(
    "CouplerMeshBackend: coupling endpoints are not registered");
}

std::any CouplerMeshBackend::Save() const
{
  return SavedFields{CopyField(left_.field.GetField()),
                     CopyField(right_.field.GetField())};
}

void CouplerMeshBackend::Restore(const std::any& checkpoint)
{
  if (!checkpoint.has_value())
    return;
  const auto& saved = std::any_cast<const SavedFields&>(checkpoint);
  RestoreField(left_.field.GetField(), saved.left);
  RestoreField(right_.field.GetField(), saved.right);
}

void CouplerMeshBackend::BeginWindow(Real, Real)
{
  window_active_ = true;
  completed_sweeps_ = 0;
}

void CouplerMeshBackend::BeginSweep(int sweep)
{
  if (!window_active_)
    throw std::logic_error(
      "CouplerMeshBackend: sweep outside a window");
  active_sweep_ = sweep;
}

void CouplerMeshBackend::EndSweep(int sweep)
{
  if (active_sweep_ != sweep)
    throw std::logic_error(
      "CouplerMeshBackend: mismatched sweep lifecycle");
  active_sweep_ = -1;
  ++completed_sweeps_;
}

void CouplerMeshBackend::CommitWindow()
{
  window_active_ = false;
  active_sweep_ = -1;
}

void CouplerMeshBackend::RejectWindow()
{
  active_sweep_ = -1;
}

Real CouplerMeshBackend::ResidualNorm(
  std::span<const Real> residual) const
{
  Real local_sum = 0.0;
  for (const Real value : residual)
    local_sum += value * value;
  Real global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                left_.mesh->comm()->get_impl());
  return std::sqrt(global_sum);
}

std::size_t CouplerMeshBackend::MeshCount() const noexcept { return 2; }
std::size_t CouplerMeshBackend::LeftOverlapDofCount() const noexcept
{
  return left_overlap_dofs_;
}
std::size_t CouplerMeshBackend::RightOverlapDofCount() const noexcept
{
  return right_overlap_dofs_;
}
int CouplerMeshBackend::CompletedSweeps() const noexcept
{
  return completed_sweeps_;
}

CouplerMeshEndpoint CouplerMeshBackend::ValidateEndpoint(
  CouplerMeshEndpoint endpoint)
{
  if (endpoint.participant == nullptr || endpoint.mesh == nullptr)
    throw std::invalid_argument(
      "CouplerMeshBackend: endpoint is incomplete");
  if (!endpoint.extract_target_interface || !endpoint.is_overlap)
    throw std::invalid_argument(
      "CouplerMeshBackend: endpoint policies are missing");
  return endpoint;
}

std::vector<Real> CouplerMeshBackend::CopyField(
  const Field<Real>& field)
{
  const auto values = field.GetDOFHolderDataHost();
  std::vector<Real> result(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    result[i] = values[i];
  return result;
}

void CouplerMeshBackend::RestoreField(
  Field<Real>& field, const std::vector<Real>& values)
{
  if (values.empty())
    return;
  field.SetDOFHolderDataHost(
    Rank1View<const Real, HostMemorySpace>(values.data(), values.size()));
}

std::size_t CouplerMeshBackend::CountOverlap(
  const CouplerMeshEndpoint& endpoint)
{
  const Omega_h::HostRead<Omega_h::Real> coordinates(
    endpoint.mesh->coords());
  std::vector<Real> coordinate(
    static_cast<std::size_t>(endpoint.mesh->dim()));
  std::size_t count = 0;
  for (int vertex = 0; vertex < endpoint.mesh->nverts(); ++vertex) {
    for (int d = 0; d < endpoint.mesh->dim(); ++d) {
      coordinate[static_cast<std::size_t>(d)] =
        coordinates[endpoint.mesh->dim() * vertex + d];
    }
    count += endpoint.is_overlap(
      std::span<const Real>(coordinate));
  }
  return count;
}

} // namespace pcms::transient
