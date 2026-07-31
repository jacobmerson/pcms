#ifndef PCMS_TRANSIENT_COUPLED_SIMULATION_HPP
#define PCMS_TRANSIENT_COUPLED_SIMULATION_HPP

#include "pcms/transient/coupling.hpp"
#include "pcms/transient/error_estimator.hpp"
#include "pcms/transient/participant.hpp"
#include "pcms/transient/schwarz.hpp"
#include "pcms/transient/timestepper.hpp"
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcms::transient
{

// Layer 3 — the time-loop driver (plan §3.4). Owns ONLY the window accept/
// reject loop; the transfer registry / redev rendezvous stays the existing
// Coupler's job (in the mesh participant adapter the driver has-a Coupler& for
// AddTransfer — omitted here since the in-process demo takes the direct
// transfer path, plan open question §10.1). Every heavy decision is an injected
// strategy: Timestepper (Δt), SchwarzIterator (sweeps), CouplingErrorEstimator.
class CoupledSimulation
{
public:
  virtual void Run(Real t_end) = 0;
  virtual SchwarzResult AdvanceWindow() = 0; // one accepted window; for tests
  virtual ~CoupledSimulation() = default;
};

// Per-window diagnostics, surfaced for the demonstration / tests.
struct WindowReport
{
  Real t_start = 0.0;
  Real dt = 0.0;
  int schwarz_iters = 0;
  Real schwarz_residual = 0.0;
  int attempts = 1; // how many proposed Δt's were tried before acceptance
  bool converged = false;
};

class TransientOverlapCoupled : public CoupledSimulation
{
public:
  // Force a rejected window when Schwarz stalls (plan §3.4): a non-converged
  // sweep cannot be trusted, so push the error above tol regardless of estimate.
  static constexpr Real kDivergePenalty = 1e3;

  TransientOverlapCoupled(std::vector<Participant*> apps, CouplingSet couplings,
                          std::unique_ptr<Timestepper> ts,
                          std::unique_ptr<SchwarzIterator> schwarz,
                          std::unique_ptr<CouplingErrorEstimator> err)
    : apps_(std::move(apps)),
      couplings_(std::move(couplings)),
      ts_(std::move(ts)),
      schwarz_(std::move(schwarz)),
      err_(std::move(err))
  {
    NegotiateCapabilities();
    // Seed the current Δt once; from here it is carried across windows and only
    // ever updated from Timestepper::Update's return (step-size contract).
    dt_ = ts_->InitialStep();
  }

  void Run(Real t_end) override
  {
    t_end_ = t_end;
    while (t_ < t_end_ - kTimeEps)
      AdvanceWindow();
  }

  SchwarzResult AdvanceWindow() override
  {
    const auto backends = UniqueBackends(couplings_);
    for (const auto& backend : backends)
      backend->BeginWindow(t_, dt_);

    // Window-start checkpoints (also the re-run point for a rejected attempt).
    std::vector<Checkpoint> ck;
    ck.reserve(apps_.size());
    for (auto* p : apps_)
      ck.push_back(p->Save());

    const Real t0 = t_;
    Real dt = std::min(dt_, t_end_ - t_); // dt_ carried across windows
    int attempts = 0;
    for (;;) {
      ++attempts;
      RestoreAll(ck);
      const SchwarzResult sr = schwarz_->Solve(apps_, couplings_, t_, dt);
      Real e = err_->Estimate(apps_, *schwarz_, couplings_, ck, t_, dt);
      if (!sr.converged)
        e = std::max(e, kDivergePenalty);

      // `dt` is the actually-integrated step (post-clamp); feed it back so the
      // controller's history is built from real steps, never a clamped-away
      // proposal (step-size contract in timestepper.hpp).
      auto [accept, dt_next] = ts_->Update(dt, e); // identical on every rank
      dt_ = dt_next;                               // carry to next attempt/window
      if (accept) {
        for (const auto& backend : backends)
          backend->CommitWindow();
        Commit();
        t_ += dt;
        last_ = WindowReport{t0, dt, sr.iters, sr.residual, attempts,
                             sr.converged};
        if (monitor_)
          monitor_(last_);
        return sr;
      }
      for (const auto& backend : backends)
        backend->RejectWindow();
      dt = std::min(dt_next, t_end_ - t_); // shrink; redo the SAME window
    }
  }

  // --- accessors for the demonstration / tests -----------------------------
  [[nodiscard]] Real CurrentTime() const noexcept { return t_; }
  [[nodiscard]] const WindowReport& LastWindow() const noexcept { return last_; }
  [[nodiscard]] const std::string& Guarantee() const noexcept
  {
    return guarantee_;
  }
  void SetMonitor(std::function<void(const WindowReport&)> m)
  {
    monitor_ = std::move(m);
  }

private:
  static constexpr Real kTimeEps = 1e-12;

  // Pick the guarantee the assembled configuration can honestly back (plan §7),
  // and assert the restart requirement the iterative Schwarz imposes (plan
  // §3.1, §9.3 capability-negotiation honesty).
  void NegotiateCapabilities()
  {
    for (auto* p : apps_) {
      if (!p->GetCapabilities().can_restart) {
        throw std::runtime_error(
          "TransientOverlapCoupled: iterative Schwarz requires a "
          "restart-capable participant, but '" +
          std::string(p->Name()) + "' reports can_restart = false");
      }
    }
    bool all_qoi = true;
    for (auto* p : apps_)
      all_qoi = all_qoi && p->GetCapabilities().reports_qoi;
    // Overlap-only visibility ⇒ we may claim coupling-converged + overlap-
    // conservative, NOT global conservation; interior QoIs let us at least
    // MONITOR it (forming the merged global field still needs Rung 2 dense
    // output + POU merge, plan Step 5).
    guarantee_ =
      all_qoi
        ? "coupling-converged; overlap-conservative; global conservation MONITORED"
        : "coupling-converged; overlap-conservative (no global-conservation claim)";
  }

  void Commit()
  {
    // In-process: nothing external to persist. In MPMD this is where committed
    // checkpoints are pushed to participants and conservation is logged/
    // monitored (plan §3.4, §7). Kept as the seam.
    ++windows_accepted_;
  }

  void RestoreAll(const std::vector<Checkpoint>& ck)
  {
    for (std::size_t i = 0; i < apps_.size(); ++i)
      apps_[i]->Restore(ck[i]);
  }

  std::vector<Participant*> apps_;
  CouplingSet couplings_;
  std::unique_ptr<Timestepper> ts_;
  std::unique_ptr<SchwarzIterator> schwarz_;
  std::unique_ptr<CouplingErrorEstimator> err_;
  std::function<void(const WindowReport&)> monitor_;
  std::string guarantee_;
  WindowReport last_;
  Real t_ = 0.0;
  Real t_end_ = 0.0;
  Real dt_ = 0.0; // current Δt; seeded from ts_->InitialStep(), carried onward
  int windows_accepted_ = 0;
};

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_COUPLED_SIMULATION_HPP
