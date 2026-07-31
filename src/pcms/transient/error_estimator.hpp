#ifndef PCMS_TRANSIENT_ERROR_ESTIMATOR_HPP
#define PCMS_TRANSIENT_ERROR_ESTIMATOR_HPP

#include "pcms/transient/coupling.hpp"
#include "pcms/transient/participant.hpp"
#include "pcms/transient/schwarz.hpp"
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

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_ERROR_ESTIMATOR_HPP
