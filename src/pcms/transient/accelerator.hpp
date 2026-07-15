#ifndef PCMS_TRANSIENT_ACCELERATOR_HPP
#define PCMS_TRANSIENT_ACCELERATOR_HPP

#include "pcms/utility/types.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pcms::transient
{

// Pure interface-space convergence acceleration (plan §3.3a). Operates on the
// flat concatenation of every coupling's target interface DOFs — the global
// Schwarz iterate x. Update maps the fixed-point evaluation G(x_in) into the
// next iterate and returns the GLOBAL residual norm ||x_out - x_in||.
//
// Collective control (plan §6): in MPMD the residual must be an Allreduce over
// the interface so every rank stops on the same iteration. That reduction
// belongs in Norm() below — the serial default is a plain 2-norm; a distributed
// build replaces the local sum with an MPI_Allreduce over the interface layout.
class InterfaceAccelerator
{
public:
  virtual void BeginWindow() = 0;
  virtual Real Update(const std::vector<Real>& x_in,
                      const std::vector<Real>& x_out,
                      std::vector<Real>& x_next) = 0;
  virtual ~InterfaceAccelerator() = default;

protected:
  // Global L2 norm of a residual vector. TODO(mpmd): replace the local sum with
  // an MPI_Allreduce so every rank tests the same value (plan §6).
  static Real Norm(const std::vector<Real>& r)
  {
    Real s = 0.0;
    for (Real v : r)
      s += v * v;
    return std::sqrt(s);
  }
};

// Fixed under-relaxation x_next = x_in + ω (x_out - x_in). ω = 1 is plain
// Picard (one-way / weak coupling); ω < 1 stabilises stronger coupling. No
// history, no tuning beyond ω. This is the plan's PicardRelaxation.
class PicardRelaxation : public InterfaceAccelerator
{
public:
  explicit PicardRelaxation(Real omega = 1.0) : omega_(omega) {}

  void BeginWindow() override {}

  Real Update(const std::vector<Real>& x_in, const std::vector<Real>& x_out,
              std::vector<Real>& x_next) override
  {
    const std::size_t n = x_in.size();
    std::vector<Real> r(n);
    for (std::size_t i = 0; i < n; ++i)
      r[i] = x_out[i] - x_in[i];
    x_next.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      x_next[i] = x_in[i] + omega_ * r[i];
    return Norm(r);
  }

private:
  Real omega_;
};

// Aitken dynamic relaxation (plan §3.3a): ω is recomputed each sweep from
// successive residuals, ω_{k+1} = -ω_k (Δr_{k-1}·(r_k-r_{k-1}))/||r_k-r_{k-1}||².
// Free, no tuning, and materially faster than Picard as the coupling stiffens
// (ρ → 1) — the property asserted in plan §9.3 "acceleration earns its keep".
class AitkenRelaxation : public InterfaceAccelerator
{
public:
  // omega_max is a large safety cap, NOT a relaxation limit: Aitken/Irons-Tuck
  // needs over-relaxation (ω ≈ 1/(1-ρ), which blows up as ρ→1), so clamping it
  // near 1 would defeat the acceleration. It only guards pathological blowup.
  explicit AitkenRelaxation(Real omega0 = 0.5, Real omega_max = 1e6)
    : omega0_(omega0), omega_(omega0), omega_max_(omega_max)
  {
  }

  void BeginWindow() override
  {
    omega_ = omega0_;
    have_prev_ = false;
  }

  Real Update(const std::vector<Real>& x_in, const std::vector<Real>& x_out,
              std::vector<Real>& x_next) override
  {
    const std::size_t n = x_in.size();
    std::vector<Real> r(n);
    for (std::size_t i = 0; i < n; ++i)
      r[i] = x_out[i] - x_in[i];
    const Real residual = Norm(r);

    if (have_prev_) {
      Real num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const Real dr = r[i] - r_prev_[i];
        num += r_prev_[i] * dr;
        den += dr * dr;
      }
      if (den > 0.0) {
        omega_ = -omega_ * num / den;
        // Keep ω in a sane band; a degenerate step falls back to under-relax.
        if (!std::isfinite(omega_) || omega_ == 0.0)
          omega_ = omega0_;
        omega_ = std::clamp(omega_, -omega_max_, omega_max_);
      }
    }

    x_next.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      x_next[i] = x_in[i] + omega_ * r[i];

    r_prev_ = std::move(r);
    have_prev_ = true;
    return residual;
  }

private:
  Real omega0_;
  Real omega_;
  Real omega_max_;
  std::vector<Real> r_prev_;
  bool have_prev_ = false;
};

// ===========================================================================
// STUB (plan §5, §3.3a): KinsolAnderson wraps KINSOL's fixed-point solver with
// Anderson acceleration, driving x = G(x) in parallel over an N_Vector on the
// interface layout. It is the plan's "big reuse" but requires SUNDIALS.
//
// To implement (gated behind PCMS_ENABLE_SUNDIALS):
//   - build a SUNContext + N_Vector over the interface DOF layout (open
//     question §10.2: confirm the N_Vector backend matches the PETSc+Kokkos
//     memory space or every sweep host-copies);
//   - set the fixed-point map to one Schwarz sweep G, m_aa = Anderson depth;
//     KINSol drives to the convergence test, replacing the loop in
//     AcceleratedSchwarz for the accelerated case;
//   - cross-check against AitkenRelaxation on the same linear fixed-point
//     (plan §9.4): both must converge the identical problem.
// Until then AitkenRelaxation is the strongest available accelerator.
// ===========================================================================
#ifdef PCMS_ENABLE_SUNDIALS
class KinsolAnderson : public InterfaceAccelerator
{
  // TODO(sundials): wrap KINSOL fixed-point + Anderson; see block comment above.
};
#endif

} // namespace pcms::transient

#endif // PCMS_TRANSIENT_ACCELERATOR_HPP
