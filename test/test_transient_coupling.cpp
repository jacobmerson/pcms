// Tier-0 unit tests for the transient coupling layer (plan §9.3): every
// assertion is a closed-form or linear-algebra answer, no application, no mesh,
// no optional dependency. Covers the orchestration loop, Schwarz iterator,
// accelerators, error/Δt control, and the capability-negotiation honesty rule.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "pcms/transient/testing/heat1d.hpp"
#include "pcms/transient/testing/linear_coupling.hpp"
#include "pcms/transient/transient.hpp"

#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

namespace tr = pcms::transient;
using pcms::Real;
using tr::testing::HeatSubdomain;
using tr::testing::LinearParticipant;
using tr::testing::Side;
using tr::testing::TimeScheme;

namespace
{

// Two LinearParticipants coupled cyclically: one Schwarz sweep is the linear
// fixed point x ← H x + b with H = [[0,g],[g,0]], spectral radius g. Returned
// pointers stay owned by `keep`.
struct LinearPair
{
  std::shared_ptr<LinearParticipant> p0, p1;
  std::vector<tr::Participant*> apps;
  tr::CouplingSet couplings;
};

// Single participant whose output feeds back into its own BC: one Schwarz sweep
// is the SCALAR linear fixed point x ← g·x + s with contraction g. The canonical
// accelerator benchmark (plan §9.1) — a single mode, so Aitken/Irons-Tuck should
// converge in a handful of sweeps where Picard needs O(log tol / log g).
struct LinearScalar
{
  std::shared_ptr<LinearParticipant> p;
  std::vector<tr::Participant*> apps;
  tr::CouplingSet couplings;
};

LinearScalar MakeLinearScalar(Real g, Real s = 1.0)
{
  LinearScalar ls;
  ls.p = std::make_shared<LinearParticipant>("p", "out", "in", g, s);
  ls.apps = {ls.p.get()};
  tr::Coupling c;
  c.source = ls.p.get();
  c.source_interface = "out";
  c.target = ls.p.get();
  c.target_interface = "in";
  ls.couplings = {c};
  return ls;
}

LinearPair MakeLinearPair(Real rho, Real s0 = 1.0, Real s1 = 2.0)
{
  LinearPair lp;
  lp.p0 = std::make_shared<LinearParticipant>("p0", "out0", "in0", rho, s0);
  lp.p1 = std::make_shared<LinearParticipant>("p1", "out1", "in1", rho, s1);
  lp.apps = {lp.p0.get(), lp.p1.get()};

  tr::Coupling a; // p0 -> p1
  a.source = lp.p0.get();
  a.source_interface = "out0";
  a.target = lp.p1.get();
  a.target_interface = "in1";
  tr::Coupling b; // p1 -> p0
  b.source = lp.p1.get();
  b.source_interface = "out1";
  b.target = lp.p0.get();
  b.target_interface = "in0";
  lp.couplings = {a, b};
  return lp;
}

} // namespace

TEST_CASE("Picard converges at the contraction rate", "[transient]")
{
  const Real rho = 0.5;
  auto lp = MakeLinearPair(rho);
  const Real tol = 1e-10;
  tr::AcceleratedSchwarz schwarz(std::make_unique<tr::PicardRelaxation>(1.0),
                                 /*max_iters=*/200, tol);
  const auto r = schwarz.Solve(lp.apps, lp.couplings, 0.0, 1.0);

  REQUIRE(r.converged);
  // Residual contracts by rho each sweep, so iters ≈ log(tol/r0)/log(rho). With
  // r0 = O(1) this is ~34; allow a generous band around the analytic estimate.
  const int predicted = static_cast<int>(std::log(tol) / std::log(rho));
  REQUIRE(r.iters >= predicted - 3);
  REQUIRE(r.iters <= predicted + 6);
}

TEST_CASE("Acceleration earns its keep on a stiff coupling", "[transient]")
{
  const Real g = 0.97; // g → 1: Picard crawls
  const Real tol = 1e-10;

  auto lp_p = MakeLinearScalar(g);
  tr::AcceleratedSchwarz picard(std::make_unique<tr::PicardRelaxation>(1.0),
                                2000, tol);
  const auto rp = picard.Solve(lp_p.apps, lp_p.couplings, 0.0, 1.0);

  auto lp_a = MakeLinearScalar(g);
  tr::AcceleratedSchwarz aitken(std::make_unique<tr::AitkenRelaxation>(), 2000,
                                tol);
  const auto ra = aitken.Solve(lp_a.apps, lp_a.couplings, 0.0, 1.0);

  REQUIRE(rp.converged);
  REQUIRE(ra.converged);
  // Aitken should reach the same tolerance in far fewer sweeps.
  REQUIRE(ra.iters * 3 < rp.iters);
}

TEST_CASE("Divergent coupling is reported and drives a Δt reject", "[transient]")
{
  SECTION("Schwarz reports non-convergence for spectral radius > 1")
  {
    auto lp = MakeLinearPair(1.2);
    tr::AcceleratedSchwarz schwarz(std::make_unique<tr::PicardRelaxation>(1.0),
                                   /*max_iters=*/30, 1e-10);
    const auto r = schwarz.Solve(lp.apps, lp.couplings, 0.0, 1.0);
    REQUIRE_FALSE(r.converged);
    REQUIRE(r.iters == 30);
  }

  SECTION("controller rejects and shrinks Δt when fed the stall penalty")
  {
    tr::AdaptiveTimestepper ts(tr::AdaptiveTimestepper::Options{});
    const Real dt = ts.InitialStep();
    auto [accept, dt_next] =
      ts.Update(dt, tr::TransientOverlapCoupled::kDivergePenalty);
    REQUIRE_FALSE(accept);
    REQUIRE(dt_next < dt);
  }
}

TEST_CASE("Timestepper sees the actually-integrated Δt, not a clamped proposal",
          "[transient]")
{
  // Step-size contract (timestepper.hpp): the orchestrator must feed Update the
  // step ACTUALLY integrated (post-clamp to t_end), not the larger value the
  // controller would have proposed. A run ending on a clamped short window is
  // the case that breaks a naive multi-step (PI/PID) controller — guard it.
  struct RecordingStepper final : tr::Timestepper
  {
    Real dt0;
    std::vector<Real>* seen;
    explicit RecordingStepper(Real d, std::vector<Real>* s) : dt0(d), seen(s) {}
    Real InitialStep() const override { return dt0; }
    std::pair<bool, Real> Update(Real dt, Real) override
    {
      seen->push_back(dt); // record the step the orchestrator integrated
      return {true, dt0};  // always accept, always suggest dt0
    }
  };

  auto ls = MakeLinearScalar(0.5);
  std::vector<Real> seen;
  const Real dt0 = 0.1, t_end = 0.25; // dt0 does not divide t_end
  tr::TransientOverlapCoupled sim(
    ls.apps, ls.couplings, std::make_unique<RecordingStepper>(dt0, &seen),
    std::make_unique<tr::AcceleratedSchwarz>(
      std::make_unique<tr::PicardRelaxation>(1.0), 200, 1e-10),
    std::make_unique<tr::NoErrorEstimate>());
  sim.Run(t_end);

  REQUIRE(seen.size() == 3);
  REQUIRE_THAT(seen[0], Catch::Matchers::WithinAbs(0.1, 1e-14));
  REQUIRE_THAT(seen[1], Catch::Matchers::WithinAbs(0.1, 1e-14));
  // Final window is clamped: the controller "wanted" 0.1 but only 0.05 fit.
  REQUIRE_THAT(seen[2], Catch::Matchers::WithinAbs(0.05, 1e-14));
  REQUIRE_THAT(sim.CurrentTime(), Catch::Matchers::WithinAbs(t_end, 1e-12));
}

TEST_CASE("Checkpoint/restore is idempotent for re-entrant advance", "[transient]")
{
  // Schwarz correctness depends entirely on Save/Restore re-entrancy (plan
  // §9.3): repeated Restore + AdvanceTo must reproduce identical output.
  HeatSubdomain sub("s", 0.0, 1.0, 21, 0.1, 5e-3, TimeScheme::BackwardEuler);
  sub.SetInitialCondition(
    [](Real x) { return std::sin(std::numbers::pi * x); });
  sub.AddProducedInterface("mid", 0.5);
  sub.AddConsumedInterface("bc", Side::Right);

  sub.SetInterface("bc", tr::InterfaceState(std::vector<Real>{0.3}));
  const auto ck = sub.Save();

  sub.AdvanceTo(0.1);
  const Real first = sub.GetInterface("mid")[0];

  for (int rep = 0; rep < 3; ++rep) {
    sub.Restore(ck);
    sub.AdvanceTo(0.1);
    REQUIRE_THAT(sub.GetInterface("mid")[0],
                 Catch::Matchers::WithinAbs(first, 1e-14));
  }
}

TEST_CASE("Capability negotiation rejects non-restartable participants",
          "[transient]")
{
  // A participant that cannot restart cannot be driven by iterative Schwarz;
  // the orchestrator must fail loudly at construction (plan §7 honesty).
  struct NonRestartable final : tr::Participant
  {
    std::string_view Name() const override { return "stuck"; }
    void AdvanceTo(Real) override {}
    tr::Checkpoint Save() const override { return {}; }
    void Restore(const tr::Checkpoint&) override {}
    tr::InterfaceState GetInterface(std::string_view) const override
    {
      return tr::InterfaceState(1);
    }
    void SetInterface(std::string_view, const tr::InterfaceState&) override {}
    tr::Capabilities GetCapabilities() const override
    {
      return tr::Capabilities{/*can_restart=*/false, false, false};
    }
  };

  NonRestartable stuck;
  std::vector<tr::Participant*> apps{&stuck};
  REQUIRE_THROWS_AS(
    tr::TransientOverlapCoupled(
      apps, {}, std::make_unique<tr::FixedTimestepper>(1e-2),
      std::make_unique<tr::AcceleratedSchwarz>(
        std::make_unique<tr::PicardRelaxation>()),
      std::make_unique<tr::NoErrorEstimate>()),
    std::runtime_error);
}

TEST_CASE("Overlapping Schwarz reproduces the monolithic heat solve",
          "[transient]")
{
  // End-to-end: two overlapping subdomains, window = substep, must match the
  // single-domain backward-Euler solution at every shared node.
  constexpr int N = 41;
  constexpr Real alpha = 0.1, h = 5e-3, t_end = 0.2;
  constexpr int iA = 15, iB = 25;
  const Real dx = 1.0 / (N - 1);
  auto X = [&](int i) { return i * dx; };
  auto ic = [](Real x) { return std::sin(std::numbers::pi * x); };

  HeatSubdomain mono("m", X(0), X(N - 1), N, alpha, h);
  mono.SetInitialCondition(ic);
  mono.AdvanceTo(t_end);
  const auto& ref = mono.Solution();

  auto left = std::make_shared<HeatSubdomain>("left", X(0), X(iB), iB + 1, alpha, h);
  auto right =
    std::make_shared<HeatSubdomain>("right", X(iA), X(N - 1), N - iA, alpha, h);
  left->SetInitialCondition(ic);
  right->SetInitialCondition(ic);
  left->AddProducedInterface("uL", X(iA));
  left->AddConsumedInterface("bcR", Side::Right);
  right->AddProducedInterface("uR", X(iB));
  right->AddConsumedInterface("bcL", Side::Left);

  tr::Coupling a{right.get(), "uR", left.get(), "bcR", tr::IdentityTransfer(), {}};
  tr::Coupling b{left.get(), "uL", right.get(), "bcL", tr::IdentityTransfer(), {}};
  std::vector<tr::Participant*> apps{left.get(), right.get()};

  tr::TransientOverlapCoupled sim(
    apps, {a, b}, std::make_unique<tr::FixedTimestepper>(h),
    std::make_unique<tr::AcceleratedSchwarz>(
      std::make_unique<tr::AitkenRelaxation>(), 100, 1e-12),
    std::make_unique<tr::NoErrorEstimate>());
  sim.Run(t_end);

  Real err = 0.0;
  const auto& uL = left->Solution();
  for (int i = 0; i <= iB; ++i)
    err = std::max(err, std::abs(uL[i] - ref[i]));
  const auto& uR = right->Solution();
  for (int j = 0; j < N - iA; ++j)
    err = std::max(err, std::abs(uR[j] - ref[iA + j]));
  REQUIRE(err < 1e-9);
}

#ifdef PCMS_ENABLE_SUNDIALS
// Plan §9.4: the SUNDIALS-backed controller must agree with the hand-rolled one.
// We cross-check both directly (accept/grow/shrink) and end-to-end on the
// overlapping-heat problem, and confirm Reset clears a multi-step controller's
// history. Built only when PCMS_ENABLE_SUNDIALS is on (unit_tests links
// pcms::transient, which carries the flag + SUNDIALS::core).
namespace
{

// Run the adaptive overlapping-heat problem with the given controller; return
// the max error vs the monolith and the number of accepted windows.
struct AdaptiveRun
{
  Real err;
  int windows;
};

AdaptiveRun RunHeatAdaptive(std::unique_ptr<tr::Timestepper> ts, Real tol)
{
  constexpr int N = 41;
  constexpr Real alpha = 0.1, h = 5e-3, t_end = 0.2;
  constexpr int iA = 15, iB = 25;
  const Real dx = 1.0 / (N - 1);
  auto X = [&](int i) { return i * dx; };
  auto ic = [](Real x) { return std::sin(std::numbers::pi * x); };

  HeatSubdomain mono("m", X(0), X(N - 1), N, alpha, h);
  mono.SetInitialCondition(ic);
  mono.AdvanceTo(t_end);
  const auto ref = mono.Solution();

  auto left = std::make_shared<HeatSubdomain>("left", X(0), X(iB), iB + 1, alpha, h);
  auto right =
    std::make_shared<HeatSubdomain>("right", X(iA), X(N - 1), N - iA, alpha, h);
  left->SetInitialCondition(ic);
  right->SetInitialCondition(ic);
  left->AddProducedInterface("uL", X(iA));
  left->AddConsumedInterface("bcR", Side::Right);
  right->AddProducedInterface("uR", X(iB));
  right->AddConsumedInterface("bcL", Side::Left);

  tr::Coupling a{right.get(), "uR", left.get(), "bcR", tr::IdentityTransfer(), {}};
  tr::Coupling b{left.get(), "uL", right.get(), "bcL", tr::IdentityTransfer(), {}};
  std::vector<tr::Participant*> apps{left.get(), right.get()};

  tr::TransientOverlapCoupled sim(
    apps, {a, b}, std::move(ts),
    std::make_unique<tr::AcceleratedSchwarz>(
      std::make_unique<tr::AitkenRelaxation>(), 100, 1e-11),
    std::make_unique<tr::StepDoublingEstimator>(tol, /*order=*/1));
  int windows = 0;
  sim.SetMonitor([&](const tr::WindowReport&) { ++windows; });
  sim.Run(t_end);

  Real err = 0.0;
  const auto& uL = left->Solution();
  for (int i = 0; i <= iB; ++i)
    err = std::max(err, std::abs(uL[i] - ref[i]));
  const auto& uR = right->Solution();
  for (int j = 0; j < N - iA; ++j)
    err = std::max(err, std::abs(uR[j] - ref[iA + j]));
  return {err, windows};
}

} // namespace

TEST_CASE("SundialsTimestepper agrees with the internal controller (accept/grow/shrink)",
          "[transient][sundials]")
{
  tr::SundialsTimestepper::Options so;
  so.dt0 = 1e-2;
  so.order = 1;
  so.controller = tr::SundialsTimestepper::Controller::I;
  tr::SundialsTimestepper sun(so);

  tr::AdaptiveTimestepper::Options ao;
  ao.dt0 = 1e-2;
  ao.order = 1;
  tr::AdaptiveTimestepper adp(ao);

  REQUIRE_THAT(sun.InitialStep(),
               Catch::Matchers::WithinAbs(adp.InitialStep(), 1e-14));

  // err > 1 ⇒ both reject and shrink.
  {
    auto [a_ok, a_dt] = adp.Update(1e-2, 4.0);
    auto [s_ok, s_dt] = sun.Update(1e-2, 4.0);
    REQUIRE_FALSE(a_ok);
    REQUIRE_FALSE(s_ok);
    REQUIRE(a_dt < 1e-2);
    REQUIRE(s_dt < 1e-2);
  }
  // err < 1 ⇒ both accept and grow (I-controller is stateless, so order-free).
  {
    auto [a_ok, a_dt] = adp.Update(1e-2, 0.25);
    auto [s_ok, s_dt] = sun.Update(1e-2, 0.25);
    REQUIRE(a_ok);
    REQUIRE(s_ok);
    REQUIRE(a_dt > 1e-2);
    REQUIRE(s_dt > 1e-2);
  }
}

TEST_CASE("SundialsTimestepper::Reset clears multi-step controller history",
          "[transient][sundials]")
{
  tr::SundialsTimestepper::Options o;
  o.dt0 = 1e-2;
  o.order = 1;
  o.controller = tr::SundialsTimestepper::Controller::PI; // has history
  tr::SundialsTimestepper ts(o);

  const auto fresh = ts.Update(1e-2, 0.5).second; // first step: empty history
  ts.Update(fresh, 0.9);                          // pollute history
  ts.Update(2e-2, 0.3);
  ts.Reset();
  const auto after = ts.Update(1e-2, 0.5).second; // must match the fresh step
  REQUIRE_THAT(after, Catch::Matchers::WithinRel(fresh, 1e-10));
}

TEST_CASE("SundialsTimestepper tracks the internal controller on the heat problem",
          "[transient][sundials]")
{
  const Real tol = 1e-3;
  const Real safety = 0.9;

  tr::AdaptiveTimestepper::Options ao;
  ao.dt0 = 5e-3;
  ao.dt_min = 5e-3;
  ao.dt_max = 0.05;
  ao.safety = safety;
  ao.order = 1;
  const auto internal =
    RunHeatAdaptive(std::make_unique<tr::AdaptiveTimestepper>(ao), tol);

  // Match the SUNDIALS I-controller to the internal one: it controls on
  // (bias·dsm), so bias = safety^{-(p+1)} reproduces dt·safety·dsm^{-1/(p+1)}.
  tr::SundialsTimestepper::Options so;
  so.dt0 = 5e-3;
  so.dt_min = 5e-3;
  so.dt_max = 0.05;
  so.order = 1;
  so.controller = tr::SundialsTimestepper::Controller::I;
  so.error_bias = std::pow(safety, -(so.order + 1));
  const auto sundials =
    RunHeatAdaptive(std::make_unique<tr::SundialsTimestepper>(so), tol);

  // Both control the coupling error to ~tol...
  REQUIRE(internal.err < 10 * tol);
  REQUIRE(sundials.err < 10 * tol);
  // ...and take a comparable number of accepted windows (not bit-identical: the
  // internal controller caps per-step growth, SUNDIALS clamps only dt_max).
  REQUIRE(sundials.windows > 0);
  REQUIRE(sundials.windows <= 2 * internal.windows);
  REQUIRE(internal.windows <= 2 * sundials.windows);
}
#endif // PCMS_ENABLE_SUNDIALS
