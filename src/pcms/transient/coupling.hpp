#ifndef PCMS_TRANSIENT_COUPLING_HPP
#define PCMS_TRANSIENT_COUPLING_HPP

#include "pcms/transient/participant.hpp"
#include <functional>
#include <string>
#include <utility>

namespace pcms::transient
{

// Partition-of-unity weight for the overlap (plan §7). Forming a global field
// from overlapping subdomain solutions needs u = Σ_i χ_i u_i or the overlap is
// double-counted; the weight is folded into a conservative merge. This first
// draft keeps χ = 1 (the current transfer behaviour) and threads the weight
// through so the honest-conservation merge (plan Step 5) can drop in later.
struct PartitionOfUnity
{
  Real weight = 1.0;
};

// The geometry the coupler actually sees for a coupling (plan §3.2). In the
// mesh deployment this carries a ClassificationRegion from the redev partition;
// for the in-process demo the overlap is implicit in the two subdomains and we
// only need the POU weight.
struct Overlap
{
  PartitionOfUnity chi{};
};

// A per-coupling transfer applied to one interface iterate. In the redev
// deployment this is a TransferHandle from Coupler::AddTransfer, and Apply is
//   src.BeginSendPhase(); srcHandle.Send(); src.EndSendPhase();
//   handle.Run();                       // both function spaces local (plan §4)
//   tgt.BeginReceivePhase(); tgtHandle.Receive(); tgt.EndReceivePhase();
// invoked once per Schwarz sweep. On the direct in-process path both interface
// states are host vectors, so the operator is a plain callable; the default is
// identity (the source already sampled onto the shared interface). The transfer
// *method* (interpolation / conservative / MC) remains the swappable axis it is
// today — here it enters as this functor rather than the method:: recipe.
using InterfaceTransfer = std::function<InterfaceState(const InterfaceState&)>;

inline InterfaceTransfer IdentityTransfer()
{
  return [](const InterfaceState& s) { return s; };
}

// Binds two participants' named interfaces through a transfer. The transient
// layer adds *when and how often* the transfer fires (inside the Schwarz loop),
// not *how* it is built.
struct Coupling
{
  Participant* source = nullptr;
  std::string source_interface;
  Participant* target = nullptr;
  std::string target_interface;
  InterfaceTransfer transfer = IdentityTransfer();
  Overlap overlap{};

  // One application of this coupling: read the source's produced interface,
  // transfer it, and return the value to impose on the target. Kept as a method
  // (rather than doing the SetInterface here) so the Schwarz iterator can gather
  // the full G(x) before accelerating.
  [[nodiscard]] InterfaceState Apply() const
  {
    return transfer(source->GetInterface(source_interface));
  }
};

using CouplingSet = std::vector<Coupling>;

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_COUPLING_HPP
