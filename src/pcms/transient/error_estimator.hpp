#ifndef PCMS_TRANSIENT_ERROR_ESTIMATOR_HPP
#define PCMS_TRANSIENT_ERROR_ESTIMATOR_HPP

#include "pcms/transient/coupling.hpp"
#include "pcms/transient/participant.hpp"
#include "pcms/transient/schwarz.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace pcms::transient
{

// Layer 2 — the coupling-error scalar the Δt controller consumes (plan §3.3c,
// §5). This estimate is PCMS-owned: SUNDIALS has no concept of "coupling
// error". The value returned is NORMALIZED to the tolerance — accept the window
// when it is ≤ 1.
//
// Contract: called with participants at the coarse end-of-window state left by
// SchwarzIterator::Solve; may re-integrate from `window_start`; MUST leave the
// participants at the state to be committed for this window.
class CouplingErrorEstimator
{
public:
  virtual Real Estimate(std::span<Participant* const> apps,
                        SchwarzIterator& schwarz, const CouplingSet& couplings,
                        const std::vector<Checkpoint>& window_start, Real t,
                        Real dt) = 0;
  virtual ~CouplingErrorEstimator() = default;
};

namespace detail
{
// Compared quantity of interest: interior scalars when a participant reports
// them (plan Rung 1), else the interface transmission data. Concatenated across
// all participants / couplings so step-doubling compares like with like.
inline std::vector<Real> GatherQoI(std::span<Participant* const> apps,
                                   const CouplingSet& couplings)
{
  std::vector<Real> q;
  bool any_qoi = false;
  for (auto* p : apps) {
    auto s = p->ReportQoI();
    if (!s.empty())
      any_qoi = true;
    q.insert(q.end(), s.begin(), s.end());
  }
  if (any_qoi)
    return q;
  // Fallback: the interface iterate.
  for (const auto& c : couplings) {
    const InterfaceState s = c.Apply();
    q.insert(q.end(), s.Data().begin(), s.Data().end());
  }
  return q;
}

inline Real L2Diff(const std::vector<Real>& a, const std::vector<Real>& b)
{
  const std::size_t n = std::min(a.size(), b.size());
  Real s = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}
} // namespace detail

// Baseline estimator for fixed-Δt runs: reports zero error and leaves the
// coarse solve committed. Pairs with FixedTimestepper (which ignores the error)
// for the plan's Step 1 / CI baseline.
class NoErrorEstimate : public CouplingErrorEstimator
{
public:
  Real Estimate(std::span<Participant* const>, SchwarzIterator&,
                const CouplingSet&, const std::vector<Checkpoint>&, Real,
                Real) override
  {
    return 0.0;
  }
};

// Rigorous ~3× coupling-error estimate by step doubling (plan §3.3c, §7): the
// window is advanced once at dt and once as two dt/2 sub-windows; the QoI
// difference is a Richardson estimate of the leading-order coupling error. The
// finer (two-half-step) solution is the one committed. `order` is the COUPLING
// scheme's temporal order (plan open question §10.3), not any code's internal
// order — first-order for a single Schwarz-converged window march.
class StepDoublingEstimator : public CouplingErrorEstimator
{
public:
  explicit StepDoublingEstimator(Real tol, int order = 1)
    : tol_(tol), order_(order)
  {
  }

  Real Estimate(std::span<Participant* const> apps, SchwarzIterator& schwarz,
                const CouplingSet& couplings,
                const std::vector<Checkpoint>& window_start, Real t,
                Real dt) override
  {
    // Participants are at the coarse (single-dt) solution on entry.
    const std::vector<Real> coarse = detail::GatherQoI(apps, couplings);

    // Fine solve: two dt/2 sub-windows from the window start.
    for (std::size_t i = 0; i < apps.size(); ++i)
      apps[i]->Restore(window_start[i]);
    schwarz.Solve(apps, couplings, t, 0.5 * dt);
    schwarz.Solve(apps, couplings, t + 0.5 * dt, 0.5 * dt);
    const std::vector<Real> fine = detail::GatherQoI(apps, couplings);

    // Leaves the (more accurate) fine solution committed.
    const Real denom = std::pow(2.0, order_) - 1.0;
    const Real err = detail::L2Diff(coarse, fine) / (denom > 0 ? denom : 1.0);
    return err / tol_;
  }

private:
  Real tol_;
  int order_;
};

// ===========================================================================
// STUB (plan §3.3c, §5, §7): OverlapResidualEstimator controls the
// splitting/coupling error CHEAPLY (~free) from the continuous interface-
// condition residual, reusing the Schwarz residual — but it needs participant
// dense output (Capabilities::has_dense_output, plan Rung 2). To implement:
// evaluate the transmission-condition mismatch on the dense-output solution at
// interior points of the window and normalise to tol. Delegates each code's
// interior temporal error to that code's own stepper (plan §7 "whose error"
// discipline) — must not be presented as the lumped temporal error that
// StepDoublingEstimator controls.
// ===========================================================================

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_ERROR_ESTIMATOR_HPP
