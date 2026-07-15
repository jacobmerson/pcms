#ifndef PCMS_TRANSIENT_SCHWARZ_HPP
#define PCMS_TRANSIENT_SCHWARZ_HPP

#include "pcms/transient/accelerator.hpp"
#include "pcms/transient/coupling.hpp"
#include "pcms/transient/participant.hpp"
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace pcms::transient
{

struct SchwarzResult
{
  int iters = 0;
  Real residual = 0.0;
  bool converged = false;
};

// Layer 2 — one coupling window's Schwarz sweep loop (plan §3.3b, §4). Drives
// the EXISTING comm + transfer path repeatedly within a window instead of once:
// impose the current interface iterate as each participant's BC, re-advance
// from the window-start checkpoint, gather G(x), accelerate, test convergence.
class SchwarzIterator
{
public:
  virtual SchwarzResult Solve(std::span<Participant* const> apps,
                              const CouplingSet& couplings, Real t, Real dt) = 0;
  virtual ~SchwarzIterator() = default;
};

// Accelerated additive (Jacobi) Schwarz. Every participant is re-integrated
// from the same window-start state each sweep using the current iterate, so the
// sweep is a fixed-point map x ← G(x); the injected InterfaceAccelerator turns
// that into x_next and reports the residual. Requires Participant::can_restart
// (asserted by the orchestrator's capability negotiation).
class AcceleratedSchwarz : public SchwarzIterator
{
public:
  AcceleratedSchwarz(std::unique_ptr<InterfaceAccelerator> accel,
                     int max_iters = 50, Real tol = 1e-10)
    : accel_(std::move(accel)), max_iters_(max_iters), tol_(tol)
  {
  }

  SchwarzResult Solve(std::span<Participant* const> apps,
                      const CouplingSet& couplings, Real t, Real dt) override
  {
    const auto offsets = Offsets(couplings);
    const std::size_t dim = offsets.back();

    accel_->BeginWindow();

    // Window-start checkpoints — the re-entrancy Schwarz depends on (plan §3.1).
    std::vector<Checkpoint> ck;
    ck.reserve(apps.size());
    for (auto* p : apps)
      ck.push_back(p->Save());

    // Initial iterate: G evaluated at the start-of-window solution (a free warm
    // start; for a fresh window these are the previous window's converged BCs).
    std::vector<Real> x = Gather(couplings, dim, offsets);

    SchwarzResult res;
    res.residual = std::numeric_limits<Real>::infinity();
    std::vector<Real> x_out, x_next;

    for (int it = 0; it < max_iters_; ++it) {
      Restore(apps, ck);
      Impose(couplings, x, offsets);
      for (auto* p : apps)
        p->AdvanceTo(t + dt);

      x_out = Gather(couplings, dim, offsets); // G(x)
      const Real r = accel_->Update(x, x_out, x_next);
      res.iters = it + 1;
      res.residual = r;
      if (r <= tol_) {
        res.converged = true;
        break; // participants hold the solution consistent with BC = x
      }
      x.swap(x_next);
    }
    // On max_iters the participants hold the last (unconverged) iterate; the
    // orchestrator rejects the window via the stall penalty (plan §3.4).
    return res;
  }

private:
  static std::vector<std::size_t> Offsets(const CouplingSet& couplings)
  {
    std::vector<std::size_t> off(couplings.size() + 1, 0);
    for (std::size_t c = 0; c < couplings.size(); ++c) {
      const auto& cp = couplings[c];
      off[c + 1] =
        off[c] + cp.source->GetInterface(cp.source_interface).Size();
    }
    return off;
  }

  // G(x): apply every coupling and concatenate the produced target interfaces.
  static std::vector<Real> Gather(const CouplingSet& couplings, std::size_t dim,
                                  const std::vector<std::size_t>& off)
  {
    std::vector<Real> x(dim);
    for (std::size_t c = 0; c < couplings.size(); ++c) {
      const InterfaceState s = couplings[c].Apply();
      for (std::size_t i = 0; i < s.Size(); ++i)
        x[off[c] + i] = s[i];
    }
    return x;
  }

  // Split the flat iterate back into per-coupling interface BCs on the targets.
  static void Impose(const CouplingSet& couplings, const std::vector<Real>& x,
                     const std::vector<std::size_t>& off)
  {
    for (std::size_t c = 0; c < couplings.size(); ++c) {
      const std::size_t n = off[c + 1] - off[c];
      InterfaceState s(n);
      for (std::size_t i = 0; i < n; ++i)
        s[i] = x[off[c] + i];
      couplings[c].target->SetInterface(couplings[c].target_interface, s);
    }
  }

  static void Restore(std::span<Participant* const> apps,
                      const std::vector<Checkpoint>& ck)
  {
    for (std::size_t i = 0; i < apps.size(); ++i)
      apps[i]->Restore(ck[i]);
  }

  std::unique_ptr<InterfaceAccelerator> accel_;
  int max_iters_;
  Real tol_;
};

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_SCHWARZ_HPP
