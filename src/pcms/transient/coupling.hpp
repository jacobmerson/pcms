#ifndef PCMS_TRANSIENT_COUPLING_HPP
#define PCMS_TRANSIENT_COUPLING_HPP

#include "pcms/transient/participant.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
// mesh participant adapter this carries a ClassificationRegion from the redev
// partition;
// for the in-process demo the overlap is implicit in the two subdomains and we
// only need the POU weight.
struct Overlap
{
  PartitionOfUnity chi{};
};

// A per-coupling transfer applied to one interface iterate. In the redev
// participant adapter this is a TransferHandle from Coupler::AddTransfer, and
// Apply is
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

// Deployment-specific data movement for a coupling. The transient layer owns
// ordering and fixed-point iteration, while a backend may implement the actual
// transfer directly, through a coupler-owned mesh, or through an MPMD
// communication path. Mesh and transport types intentionally stay out of this
// interface so pcms::transient remains dependency-free.
class CouplingBackend
{
public:
  [[nodiscard]] virtual InterfaceState Transfer(
    Participant& source, std::string_view source_interface,
    Participant& target, std::string_view target_interface,
    const InterfaceTransfer& transfer) = 0;

  // Coupler-side fields are part of the re-entrant Schwarz state. Immutable
  // meshes, layouts, and localization structures need not be checkpointed.
  [[nodiscard]] virtual std::any Save() const { return {}; }
  virtual void Restore(const std::any&) {}

  virtual void BeginWindow(Real, Real) {}
  virtual void BeginSweep(int) {}
  virtual void EndSweep(int) {}
  virtual void CommitWindow() {}
  virtual void RejectWindow() {}

  // Distributed backends override this with the collective norm used by every
  // rank to make the same convergence decision.
  [[nodiscard]] virtual Real ResidualNorm(
    std::span<const Real> residual) const
  {
    Real sum = 0.0;
    for (Real value : residual)
      sum += value * value;
    return std::sqrt(sum);
  }

  virtual ~CouplingBackend() = default;
};

// The original in-process behavior, retained as the default backend.
class DirectCouplingBackend final : public CouplingBackend
{
public:
  [[nodiscard]] InterfaceState Transfer(
    Participant& source, std::string_view source_interface, Participant&,
    std::string_view, const InterfaceTransfer& transfer) override
  {
    return transfer(source.GetInterface(source_interface));
  }
};

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
  std::shared_ptr<CouplingBackend> backend;

  // One application of this coupling: read the source's produced interface,
  // transfer it, and return the value to impose on the target. Kept as a method
  // (rather than doing the SetInterface here) so the Schwarz iterator can gather
  // the full G(x) before accelerating.
  [[nodiscard]] InterfaceState Apply() const
  {
    if (source == nullptr || target == nullptr)
      throw std::invalid_argument("Coupling: null participant");
    if (backend) {
      return backend->Transfer(*source, source_interface, *target,
                               target_interface, transfer);
    }
    return transfer(source->GetInterface(source_interface));
  }
};

using CouplingSet = std::vector<Coupling>;

inline std::vector<std::shared_ptr<CouplingBackend>> UniqueBackends(
  const CouplingSet& couplings)
{
  std::vector<std::shared_ptr<CouplingBackend>> result;
  for (const Coupling& coupling : couplings) {
    if (!coupling.backend)
      continue;
    if (std::find(result.begin(), result.end(), coupling.backend) ==
        result.end()) {
      result.push_back(coupling.backend);
    }
  }
  return result;
}

// Express one multiplicative (sequential) Schwarz sweep as a fixed-point map.
//
// The flat state contains one transferred interface value for every Coupling.
// Participants are advanced in `apps` order. An incoming coupling whose source
// has already advanced in the current sweep uses that source's new value;
// otherwise it uses the value supplied in the input state. Consequently, for
// participants A then B, B immediately consumes A's new interface while A
// consumes B's value from the previous sweep. This is Gauss-Seidel/sequential
// Schwarz, exposed as x_out = G(x) so the normal fixed-point accelerators can
// be applied without changing the underlying algorithm.
class SequentialCouplingFixedPoint
{
public:
  explicit SequentialCouplingFixedPoint(const CouplingSet& couplings)
    : couplings_(couplings), offsets_(couplings.size() + 1, 0)
  {
    for (std::size_t c = 0; c < couplings_.size(); ++c) {
      Validate(couplings_[c]);
      offsets_[c + 1] =
        offsets_[c] + couplings_[c].Apply().Size();
    }
  }

  [[nodiscard]] std::vector<Real> Gather() const
  {
    std::vector<Real> state(offsets_.back());
    for (std::size_t c = 0; c < couplings_.size(); ++c) {
      Store(c, couplings_[c].Apply(), state);
    }
    return state;
  }

  [[nodiscard]] std::vector<Real> Evaluate(
    std::span<Participant* const> apps, std::span<const Real> state,
    Real target_time, int sweep = 0) const
  {
    if (state.size() != offsets_.back()) {
      throw std::invalid_argument(
        "SequentialCouplingFixedPoint: interface state size mismatch");
    }
    for (const Coupling& coupling : couplings_) {
      if (std::find(apps.begin(), apps.end(), coupling.source) == apps.end() ||
          std::find(apps.begin(), apps.end(), coupling.target) == apps.end()) {
        throw std::invalid_argument(
          "SequentialCouplingFixedPoint: coupling references a participant "
          "outside the sweep");
      }
    }

    std::vector<Participant*> advanced;
    advanced.reserve(apps.size());
    const auto backends = UniqueBackends(couplings_);
    for (const auto& backend : backends)
      backend->BeginSweep(sweep);

    try {
      for (Participant* app : apps) {
        if (app == nullptr) {
          throw std::invalid_argument(
            "SequentialCouplingFixedPoint: null participant");
        }

        for (std::size_t c = 0; c < couplings_.size(); ++c) {
          const Coupling& coupling = couplings_[c];
          if (coupling.target != app)
            continue;

          const bool source_has_advanced =
            std::find(advanced.begin(), advanced.end(), coupling.source) !=
            advanced.end();
          InterfaceState incoming =
            source_has_advanced ? coupling.Apply() : Load(c, state);
          app->SetInterface(coupling.target_interface, incoming);
        }

        app->AdvanceTo(target_time);
        advanced.push_back(app);
      }

      std::vector<Real> result = Gather();
      for (const auto& backend : backends)
        backend->EndSweep(sweep);
      return result;
    } catch (...) {
      for (const auto& backend : backends)
        backend->EndSweep(sweep);
      throw;
    }
  }

private:
  static void Validate(const Coupling& coupling)
  {
    if (coupling.source == nullptr || coupling.target == nullptr) {
      throw std::invalid_argument(
        "SequentialCouplingFixedPoint: coupling has a null participant");
    }
  }

  [[nodiscard]] InterfaceState Load(
    std::size_t coupling, std::span<const Real> state) const
  {
    InterfaceState result(offsets_[coupling + 1] - offsets_[coupling]);
    for (std::size_t i = 0; i < result.Size(); ++i)
      result[i] = state[offsets_[coupling] + i];
    return result;
  }

  void Store(std::size_t coupling, const InterfaceState& value,
             std::span<Real> state) const
  {
    const std::size_t expected =
      offsets_[coupling + 1] - offsets_[coupling];
    if (value.Size() != expected) {
      throw std::runtime_error(
        "SequentialCouplingFixedPoint: interface size changed during sweep");
    }
    for (std::size_t i = 0; i < value.Size(); ++i)
      state[offsets_[coupling] + i] = value[i];
  }

  const CouplingSet& couplings_;
  std::vector<std::size_t> offsets_;
};

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_COUPLING_HPP
