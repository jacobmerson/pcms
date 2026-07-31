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

// Accelerated multiplicative (Gauss-Seidel) Schwarz. The coupling graph turns
// each ordered sequential sweep into x_out = G_seq(x); the accelerator
// contract relaxes or accelerates that interface fixed point.
class AcceleratedSequentialSchwarz : public SchwarzIterator
{
public:
  AcceleratedSequentialSchwarz(std::unique_ptr<InterfaceAccelerator> accel,
                               int max_iters = 50, Real tol = 1e-10)
    : accel_(std::move(accel)), max_iters_(max_iters), tol_(tol)
  {
  }

  SchwarzResult Solve(std::span<Participant* const> apps,
                      const CouplingSet& couplings, Real t, Real dt) override
  {
    SequentialCouplingFixedPoint fixed_point(couplings);
    accel_->BeginWindow();
    const auto backends = UniqueBackends(couplings);

    std::vector<Checkpoint> checkpoints;
    checkpoints.reserve(apps.size());
    for (Participant* app : apps)
      checkpoints.push_back(app->Save());
    std::vector<std::any> backend_checkpoints;
    backend_checkpoints.reserve(backends.size());
    for (const auto& backend : backends)
      backend_checkpoints.push_back(backend->Save());

    std::vector<Real> x = fixed_point.Gather();
    std::vector<Real> x_out;
    std::vector<Real> x_next;

    SchwarzResult result;
    result.residual = std::numeric_limits<Real>::infinity();

    for (int iteration = 0; iteration < max_iters_; ++iteration) {
      Restore(apps, checkpoints);
      for (std::size_t i = 0; i < backends.size(); ++i)
        backends[i]->Restore(backend_checkpoints[i]);
      x_out = fixed_point.Evaluate(apps, x, t + dt, iteration);

      result.residual = accel_->Update(x, x_out, x_next);
      if (!backends.empty()) {
        std::vector<Real> residual(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
          residual[i] = x_out[i] - x[i];
        result.residual = 0.0;
        for (const auto& backend : backends) {
          result.residual =
            std::max(result.residual, backend->ResidualNorm(residual));
        }
      }
      result.iters = iteration + 1;
      if (result.residual <= tol_) {
        result.converged = true;
        break;
      }
      x.swap(x_next);
    }

    return result;
  }

private:
  static void Restore(std::span<Participant* const> apps,
                      const std::vector<Checkpoint>& checkpoints)
  {
    for (std::size_t i = 0; i < apps.size(); ++i)
      apps[i]->Restore(checkpoints[i]);
  }

  std::unique_ptr<InterfaceAccelerator> accel_;
  int max_iters_;
  Real tol_;
};

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_SCHWARZ_HPP
