// Demonstration: transient overlapping-Schwarz coupling of a PARTITIONED domain
// with OVERLAP, built entirely on the transient layer
// (src/pcms/transient, docs/transient_coupled_simulation_plan.md).
//
// The 1D heat equation u_t = α u_xx on [0,1] with u(0)=u(1)=0 and IC sin(πx) is
// decomposed into two OVERLAPPING subdomains:
//
//     0 ────────────── xA ========== xB ────────────── 1
//     |     left-only    |  overlap   |    right-only   |
//     └────── Ω_L = [0, xB] ──────────┘
//                        └─────── Ω_R = [xA, 1] ────────┘
//
// Each subdomain is a black-box HeatSubdomain participant (backward Euler,
// internal subcycling). They are coupled by exchanging the neighbour's solution
// at their artificial interface (a Dirichlet transmission condition) and
// Schwarz-iterated to consistency within every coupling window. At convergence
// with window = one substep, the coupled solution reproduces the MONOLITHIC
// single-domain backward-Euler solve — that is the correctness check. The
// adaptive-Δt run then advances with larger windows + internal subcycling under
// step-doubling error control and stays within tolerance of the monolith.
//
// No SUNDIALS / PETSc / MFEM / redev: the whole control stack (Schwarz,
// accelerator, error estimator, Δt controller) runs on the dependency-free
// internal implementations (plan Tier 0 / §9.4).

#include "pcms/transient/testing/heat1d.hpp"
#include "pcms/transient/transient.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <vector>

namespace tr = pcms::transient;
using tr::testing::HeatSubdomain;
using tr::testing::Side;
using tr::testing::TimeScheme;
using pcms::Real;

namespace
{

// ---- global problem definition (shared by monolith and the two subdomains) --
constexpr int kNGlobal = 41;         // global nodes: x_i = i/40
constexpr Real kAlpha = 0.1;         // diffusivity
constexpr Real kTEnd = 0.5;          // final time
constexpr Real kSubstep = 5e-3;      // internal backward-Euler substep h
constexpr int kIA = 15;              // overlap left index  -> xA = 0.375
constexpr int kIB = 25;              // overlap right index -> xB = 0.625
constexpr Real kDx = 1.0 / (kNGlobal - 1);

Real InitialCondition(Real x) { return std::sin(std::numbers::pi * x); }

Real NodeX(int i) { return static_cast<Real>(i) * kDx; }

// Monolithic reference: one subdomain over the whole domain, both ends physical.
std::vector<Real> MonolithReference()
{
  HeatSubdomain full("monolith", NodeX(0), NodeX(kNGlobal - 1), kNGlobal, kAlpha,
                     kSubstep, TimeScheme::BackwardEuler);
  full.SetInitialCondition(InitialCondition);
  full.AdvanceTo(kTEnd);
  return full.Solution();
}

struct Subdomains
{
  std::shared_ptr<HeatSubdomain> left;
  std::shared_ptr<HeatSubdomain> right;
};

// Build the two overlapping subdomains with their produced/consumed interfaces.
Subdomains MakeSubdomains()
{
  const Real xA = NodeX(kIA), xB = NodeX(kIB);
  auto left = std::make_shared<HeatSubdomain>(
    "left", NodeX(0), xB, kIB + 1, kAlpha, kSubstep, TimeScheme::BackwardEuler);
  auto right = std::make_shared<HeatSubdomain>(
    "right", xA, NodeX(kNGlobal - 1), kNGlobal - kIA, kAlpha, kSubstep,
    TimeScheme::BackwardEuler);
  left->SetInitialCondition(InitialCondition);
  right->SetInitialCondition(InitialCondition);

  // Left's right end (xB) is fed by the right subdomain's solution there;
  // right's left end (xA) is fed by the left subdomain's solution there.
  left->AddProducedInterface("uL_at_xA", xA);
  left->AddConsumedInterface("bc_at_xB", Side::Right);
  right->AddProducedInterface("uR_at_xB", xB);
  right->AddConsumedInterface("bc_at_xA", Side::Left);
  return {left, right};
}

tr::CouplingSet MakeCouplings(const Subdomains& s)
{
  tr::Coupling a; // right -> left, imposed at xB
  a.source = s.right.get();
  a.source_interface = "uR_at_xB";
  a.target = s.left.get();
  a.target_interface = "bc_at_xB";

  tr::Coupling b; // left -> right, imposed at xA
  b.source = s.left.get();
  b.source_interface = "uL_at_xA";
  b.target = s.right.get();
  b.target_interface = "bc_at_xA";
  return {a, b};
}

// Max |coupled - monolith| over each subdomain's nodes (they align with the
// global grid, so the sample points are exact — the transfer is a copy here).
Real MaxErrorVsMonolith(const Subdomains& s, const std::vector<Real>& ref)
{
  Real e = 0.0;
  const auto& uL = s.left->Solution();
  for (int i = 0; i <= kIB; ++i)
    e = std::max(e, std::abs(uL[i] - ref[i]));
  const auto& uR = s.right->Solution();
  for (int j = 0; j < kNGlobal - kIA; ++j)
    e = std::max(e, std::abs(uR[j] - ref[kIA + j]));
  return e;
}

bool Check(const char* what, bool ok)
{
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  return ok;
}

} // namespace

int main()
{
  bool all_ok = true;
  const std::vector<Real> ref = MonolithReference();
  std::printf("Transient overlapping-Schwarz demonstration (1D heat, α=%.3g, "
              "t_end=%.3g)\n",
              kAlpha, kTEnd);
  std::printf("  global grid: %d nodes, overlap [x%d, x%d] = [%.3f, %.3f]\n\n",
              kNGlobal, kIA, kIB, NodeX(kIA), NodeX(kIB));

  // ---------------------------------------------------------------------------
  // Run 1 — window = substep, fixed Δt, Aitken Schwarz. Must reproduce the
  // monolithic solve to Schwarz tolerance.
  // ---------------------------------------------------------------------------
  {
    auto s = MakeSubdomains();
    auto couplings = MakeCouplings(s);
    std::vector<tr::Participant*> apps{s.left.get(), s.right.get()};

    auto sim = std::make_unique<tr::TransientOverlapCoupled>(
      apps, couplings, std::make_unique<tr::FixedTimestepper>(kSubstep),
      std::make_unique<tr::AcceleratedSchwarz>(
        std::make_unique<tr::AitkenRelaxation>(), /*max_iters=*/100,
        /*tol=*/1e-12),
      std::make_unique<tr::NoErrorEstimate>());

    int max_iters = 0, windows = 0;
    Real sum_iters = 0;
    sim->SetMonitor([&](const tr::WindowReport& w) {
      max_iters = std::max(max_iters, w.schwarz_iters);
      sum_iters += w.schwarz_iters;
      ++windows;
    });
    std::printf("Run 1: fixed Δt = substep, Aitken Schwarz\n");
    std::printf("  guarantee advertised: %s\n", sim->Guarantee().c_str());
    sim->Run(kTEnd);
    const Real err = MaxErrorVsMonolith(s, ref);
    std::printf("  windows: %d, avg Schwarz iters: %.1f, max: %d\n", windows,
                sum_iters / windows, max_iters);
    std::printf("  max |coupled - monolith| = %.3e\n", err);
    all_ok &= Check("Schwarz reproduces the monolithic solve (err < 1e-9)",
                    err < 1e-9);
    std::printf("\n");
  }

  // ---------------------------------------------------------------------------
  // Run 2 — adaptive Δt (internal controller) + step-doubling coupling-error
  // estimate, larger windows with internal subcycling. Splitting error is
  // controlled to tol; result stays close to the monolith.
  // ---------------------------------------------------------------------------
  {
    auto s = MakeSubdomains();
    auto couplings = MakeCouplings(s);
    std::vector<tr::Participant*> apps{s.left.get(), s.right.get()};

    const Real tol = 1e-3;
    tr::AdaptiveTimestepper::Options opt;
    opt.dt0 = kSubstep;    // start at the substep and let the controller grow it
    opt.dt_min = kSubstep; // never below one internal substep
    opt.dt_max = 0.05;
    opt.max_factor = 2.0;  // gentle growth so the controller settles, not saws
    opt.order = 1;

    auto sim = std::make_unique<tr::TransientOverlapCoupled>(
      apps, couplings, std::make_unique<tr::AdaptiveTimestepper>(opt),
      std::make_unique<tr::AcceleratedSchwarz>(
        std::make_unique<tr::AitkenRelaxation>(), 100, 1e-11),
      std::make_unique<tr::StepDoublingEstimator>(tol, /*order=*/1));

    std::printf("Run 2: adaptive Δt (step-doubling, tol=%.0e), subcycling\n",
                tol);
    int windows = 0, rejects = 0;
    sim->SetMonitor([&](const tr::WindowReport& w) {
      ++windows;
      rejects += (w.attempts - 1);
      std::printf("  window @ t=%.4f  Δt=%.4e  schwarz_iters=%d  attempts=%d\n",
                  w.t_start, w.dt, w.schwarz_iters, w.attempts);
    });
    sim->Run(kTEnd);
    const Real err = MaxErrorVsMonolith(s, ref);
    std::printf("  accepted windows: %d, rejected attempts: %d\n", windows,
                rejects);
    std::printf("  max |coupled - monolith| = %.3e (tol=%.0e)\n", err, tol);
    // Splitting error is controlled but nonzero; allow a small multiple of tol.
    all_ok &= Check("adaptive coupling-error control keeps err ~ tol", err < 1e-2);
    std::printf("\n");
  }

#ifdef PCMS_ENABLE_SUNDIALS
  // ---------------------------------------------------------------------------
  // Run 3 — same adaptive problem driven by the SUNDIALS SUNAdaptController
  // (PI) via SundialsTimestepper, showing the real controller is a drop-in.
  // ---------------------------------------------------------------------------
  {
    auto s = MakeSubdomains();
    auto couplings = MakeCouplings(s);
    std::vector<tr::Participant*> apps{s.left.get(), s.right.get()};

    const Real tol = 1e-3;
    tr::SundialsTimestepper::Options opt;
    opt.dt0 = kSubstep;
    opt.dt_min = kSubstep;
    opt.dt_max = 0.05;
    opt.order = 1;
    opt.controller = tr::SundialsTimestepper::Controller::PI;

    auto sim = std::make_unique<tr::TransientOverlapCoupled>(
      apps, couplings, std::make_unique<tr::SundialsTimestepper>(opt),
      std::make_unique<tr::AcceleratedSchwarz>(
        std::make_unique<tr::AitkenRelaxation>(), 100, 1e-11),
      std::make_unique<tr::StepDoublingEstimator>(tol, /*order=*/1));

    std::printf("Run 3: adaptive Δt via SUNDIALS SUNAdaptController (PI, tol=%.0e)\n",
                tol);
    int windows = 0, rejects = 0;
    sim->SetMonitor([&](const tr::WindowReport& w) {
      ++windows;
      rejects += (w.attempts - 1);
    });
    sim->Run(kTEnd);
    const Real err = MaxErrorVsMonolith(s, ref);
    std::printf("  accepted windows: %d, rejected attempts: %d\n", windows,
                rejects);
    std::printf("  max |coupled - monolith| = %.3e (tol=%.0e)\n", err, tol);
    all_ok &= Check("SUNDIALS controller keeps err ~ tol", err < 1e-2);
    std::printf("\n");
  }
#endif

  std::printf("%s\n", all_ok ? "DEMONSTRATION PASSED" : "DEMONSTRATION FAILED");
  return all_ok ? 0 : 1;
}
