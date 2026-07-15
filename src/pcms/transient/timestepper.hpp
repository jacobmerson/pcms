#ifndef PCMS_TRANSIENT_TIMESTEPPER_HPP
#define PCMS_TRANSIENT_TIMESTEPPER_HPP

#include "pcms/utility/types.h"
#include <algorithm>
#include <cmath>
#include <utility>

#ifdef PCMS_ENABLE_SUNDIALS
#include <sundials/sundials_adaptcontroller.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_errors.h>
#include <sundials/sundials_types.h>
#include <sunadaptcontroller/sunadaptcontroller_soderlind.h>
#include <stdexcept>
#endif

namespace pcms::transient
{

// Layer 2 — the window-Δt feedback controller (plan §3.3d). FEEDBACK, not
// predictive: the window is advanced at some Δt and measured, then Update
// decides accept/reject and the next Δt from the normalized coupling error.
//
// Step-size CONTRACT (the reason there is no Propose(t)): the controller learns
// Δt ONLY through Update's `dt` argument, which the orchestrator guarantees is
// the step ACTUALLY integrated (post-clamping to t_end / output points), paired
// with the error measured for exactly that step. The controller must build all
// of its history — previous errors AND previous step sizes — from those
// (dt, err) pairs, never from a value it proposed. InitialStep() only seeds the
// very first window and is read once. This makes the proposed-vs-actual Δt
// desync (which bites multi-step PI/PID controllers on a clamped final window)
// unrepresentable, and maps 1:1 onto SUNDIALS SUNAdaptController, whose
// primitive is EstimateStep(h, p, dsm)->hnew with no separate "propose".
//
// Accept/reject history: within one window Update may be called several times
// (reject → reject → accept) for the SAME physical window. Only the ACCEPTED
// (dt, err) pair advances the smooth multi-step history; the controller owns the
// accept decision, so it can manage this. Reset() clears history at a modelled
// discontinuity or across reuse (maps to SUNAdaptController_Reset).
//
// Collective control (plan §6): Update must consume the GLOBALLY-reduced error
// so every replica of the controller computes an identical dt_next and the
// participants stay in lockstep. Here (serial) the error is already global.
class Timestepper
{
public:
  // First-window Δt seed; read exactly once by the orchestrator.
  [[nodiscard]] virtual Real InitialStep() const = 0;
  // {accept?, dt_next}. `dt` is the actually-integrated step; `err` is
  // normalized to tol (accept iff ≤ 1).
  virtual std::pair<bool, Real> Update(Real dt, Real err) = 0;
  // Clear any accumulated step-size / error history.
  virtual void Reset() {}
  virtual ~Timestepper() = default;
};

// Ignores the error signal: always accepts, always returns the same Δt. The
// plan's baseline / debugging / CI stepper (§3.3d) and the driver for Step 1.
class FixedTimestepper : public Timestepper
{
public:
  explicit FixedTimestepper(Real dt) : dt_(dt) {}
  Real InitialStep() const override { return dt_; }
  std::pair<bool, Real> Update(Real, Real) override { return {true, dt_}; }

private:
  Real dt_;
};

// ===========================================================================
// Internal replacement for the plan's SundialsTimestepper (plan §3.3d, §5),
// which wraps a SUNAdaptController (PID / PI / Söderlind). This is a hand-rolled
// elementary I-controller so the accept/reject and Δt-adaptation LOGIC is fully
// exercised without SUNDIALS (plan §9.4). Being pure feedback (no proposed-Δt
// state — see the Timestepper contract) it is stateless: a PI/PID drop-in adds
// history members updated in Update and cleared in Reset. When
// PCMS_ENABLE_SUNDIALS lands, SundialsTimestepper wraps SUNAdaptController
// (InitialStep→seed, Update→EstimateStep+UpdateH, Reset→Reset) and must select
// the same Δt sequence as this controller on the manufactured-error problem
// (plan §9.4 requires the two implementations to agree).
//
//   accept  ⇔  err ≤ 1
//   dt_next =  dt · clamp( safety · err^(-1/(p+1)),  min_factor, max_factor )
//
// with err the coupling error normalized to tol and p the coupling order.
// ===========================================================================
class AdaptiveTimestepper : public Timestepper
{
public:
  struct Options
  {
    Real dt0 = 1e-2;
    Real dt_min = 1e-8;
    Real dt_max = 1.0;
    Real safety = 0.9;
    Real min_factor = 0.2; // never shrink by more than 5×
    Real max_factor = 5.0; // never grow by more than 5×
    int order = 1;         // coupling scheme temporal order (plan §10.3)
  };

  AdaptiveTimestepper() : AdaptiveTimestepper(Options{}) {}
  explicit AdaptiveTimestepper(Options opt) : opt_(opt) {}

  Real InitialStep() const override { return opt_.dt0; }

  std::pair<bool, Real> Update(Real dt, Real err) override
  {
    // Guard err ≈ 0 (exact/constant transient) so the factor stays finite.
    const Real e = std::max(err, 1e-12);
    Real factor = opt_.safety * std::pow(e, -1.0 / (opt_.order + 1));
    factor = std::clamp(factor, opt_.min_factor, opt_.max_factor);

    const bool accept = err <= 1.0;
    // On reject, only shrink (factor < 1); never grow a rejected step.
    if (!accept)
      factor = std::min(factor, 1.0);

    // Derived purely from the actually-integrated `dt` (contract above); the
    // I-controller keeps no history, so Reset() is a no-op.
    const Real dt_next = std::clamp(dt * factor, opt_.dt_min, opt_.dt_max);
    return {accept, dt_next};
  }

private:
  Options opt_;
};

#ifdef PCMS_ENABLE_SUNDIALS
// The plan's real Δt controller (plan §3.3d, §5): a thin wrapper over a SUNDIALS
// SUNAdaptController. It is a drop-in for AdaptiveTimestepper under the exact
// same Design-B contract — InitialStep seeds, Update maps to
// SUNAdaptController_EstimateStep (+ UpdateH on accept), Reset maps to
// SUNAdaptController_Reset. The normalized coupling error we already produce is
// SUNDIALS' `dsm` verbatim: EstimateStep targets dsm just below 1, i.e. dsm < 1
// ⇔ accept, matching our accept test. No proposed-vs-actual Δt state exists to
// desync (that was the point of dropping Propose), so a multi-step controller
// like PI/PID/Söderlind is safe here.
class SundialsTimestepper : public Timestepper
{
public:
  enum class Controller { I, PI, PID, ExpGus, ImpGus, Soderlind };

  struct Options
  {
    Real dt0 = 1e-2;
    Real dt_min = 1e-8;
    Real dt_max = 1.0;
    int order = 1;                          // coupling scheme temporal order
    Controller controller = Controller::PI; // robust default; PID's D-term is
                                            // noisy on the coupling-error signal
    Real error_bias = -1.0;                 // < 0 ⇒ leave SUNDIALS default
  };

  SundialsTimestepper() : SundialsTimestepper(Options{}) {}
  explicit SundialsTimestepper(Options opt) : opt_(opt)
  {
    if (SUNContext_Create(SUN_COMM_NULL, &ctx_) != SUN_SUCCESS)
      throw std::runtime_error("SundialsTimestepper: SUNContext_Create failed");
    controller_ = MakeController(opt_.controller, ctx_);
    if (controller_ == nullptr) {
      SUNContext_Free(&ctx_);
      throw std::runtime_error(
        "SundialsTimestepper: SUNAdaptController creation failed");
    }
    if (opt_.error_bias > 0.0)
      SUNAdaptController_SetErrorBias(controller_, opt_.error_bias);
  }

  SundialsTimestepper(const SundialsTimestepper&) = delete;
  SundialsTimestepper& operator=(const SundialsTimestepper&) = delete;

  ~SundialsTimestepper() override
  {
    if (controller_ != nullptr)
      SUNAdaptController_Destroy(controller_);
    if (ctx_ != nullptr)
      SUNContext_Free(&ctx_);
  }

  Real InitialStep() const override { return opt_.dt0; }

  std::pair<bool, Real> Update(Real dt, Real err) override
  {
    // dsm is the tol-normalized coupling error (guard 0 so the controller's
    // 1/dsm factor stays finite on an exact/constant transient).
    const sunrealtype dsm = std::max(err, Real(1e-12));
    sunrealtype hnew = dt; // fallback if EstimateStep errors
    SUNAdaptController_EstimateStep(controller_, static_cast<sunrealtype>(dt),
                                    opt_.order, dsm, &hnew);
    const bool accept = err <= 1.0;
    if (accept) // commit only the accepted step to the controller's history
      SUNAdaptController_UpdateH(controller_, static_cast<sunrealtype>(dt), dsm);
    return {accept,
            std::clamp(static_cast<Real>(hnew), opt_.dt_min, opt_.dt_max)};
  }

  void Reset() override { SUNAdaptController_Reset(controller_); }

private:
  static SUNAdaptController MakeController(Controller c, SUNContext ctx)
  {
    switch (c) {
      case Controller::I: return SUNAdaptController_I(ctx);
      case Controller::PI: return SUNAdaptController_PI(ctx);
      case Controller::PID: return SUNAdaptController_PID(ctx);
      case Controller::ExpGus: return SUNAdaptController_ExpGus(ctx);
      case Controller::ImpGus: return SUNAdaptController_ImpGus(ctx);
      case Controller::Soderlind: return SUNAdaptController_Soderlind(ctx);
    }
    return nullptr;
  }

  Options opt_;
  SUNContext ctx_ = nullptr;
  SUNAdaptController controller_ = nullptr;
};
#endif

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_TIMESTEPPER_HPP
