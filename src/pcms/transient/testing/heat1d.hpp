#ifndef PCMS_TRANSIENT_TESTING_HEAT1D_HPP
#define PCMS_TRANSIENT_TESTING_HEAT1D_HPP

#include "pcms/transient/participant.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// In-tree demonstration participant (plan §9.1): a 1D transient-heat subdomain.
// It is a genuine black-box PDE code from the coupler's viewpoint — it owns its
// grid and time integrator, subcycles internally in AdvanceTo, and exposes only
// Save/Restore + interface transmission data. Two of these on overlapping
// intervals form the partitioned-domain-with-overlap demonstration; at Schwarz
// convergence (window = one substep) their solution reproduces the monolithic
// single-domain backward-Euler solve.
namespace pcms::transient::testing
{

enum class TimeScheme
{
  ForwardEuler, // explicit; conditionally stable (h ≤ dx²/2α)
  BackwardEuler // implicit; unconditionally stable (tridiagonal Thomas solve)
};

enum class Side
{
  Left,
  Right
};

// u_t = α u_xx on [x_left, x_right] with Dirichlet ends, uniform grid. One end
// is typically a physical BC and the other an artificial interface whose value
// is supplied each Schwarz sweep via SetInterface.
class HeatSubdomain final : public Participant
{
public:
  HeatSubdomain(std::string name, Real x_left, Real x_right, std::size_t nnodes,
                Real alpha, Real internal_dt,
                TimeScheme scheme = TimeScheme::BackwardEuler)
    : name_(std::move(name)),
      alpha_(alpha),
      internal_dt_(internal_dt),
      scheme_(scheme),
      x_(nnodes),
      u_(nnodes, 0.0)
  {
    if (nnodes < 2)
      throw std::invalid_argument("HeatSubdomain: need at least 2 nodes");
    dx_ = (x_right - x_left) / static_cast<Real>(nnodes - 1);
    for (std::size_t i = 0; i < nnodes; ++i)
      x_[i] = x_left + static_cast<Real>(i) * dx_;
    left_bc_ = 0.0;
    right_bc_ = 0.0;
  }

  // Initial condition u(x, 0) = f(x); also seeds the Dirichlet end values.
  template <typename F>
  void SetInitialCondition(F&& f)
  {
    for (std::size_t i = 0; i < x_.size(); ++i)
      u_[i] = f(x_[i]);
    left_bc_ = u_.front();
    right_bc_ = u_.back();
  }

  // Register the transmission data this subdomain PRODUCES for `name`: its
  // solution sampled at physical location `x`. (A neighbour consumes it.)
  void AddProducedInterface(std::string name, Real x) { produced_[name] = x; }

  // Register the transmission BC this subdomain CONSUMES for `name`: the value
  // is imposed at the given end each sweep.
  void AddConsumedInterface(std::string name, Side side)
  {
    consumed_[name] = side;
  }

  // --- Participant ---------------------------------------------------------
  std::string_view Name() const override { return name_; }

  void AdvanceTo(Real t_target) override
  {
    if (t_target <= time_)
      return;
    // Black-box internal subcycling: split [time_, t_target] into whole
    // substeps of ~internal_dt_ with the current interface BCs held constant.
    const Real span = t_target - time_;
    const int nsub =
      std::max(1, static_cast<int>(std::ceil(span / internal_dt_ - 1e-12)));
    const Real h = span / static_cast<Real>(nsub);
    for (int s = 0; s < nsub; ++s)
      Step(h);
    time_ = t_target;
  }

  Checkpoint Save() const override
  {
    State st{u_, left_bc_, right_bc_};
    return Checkpoint{time_, std::move(st)};
  }

  void Restore(const Checkpoint& ck) override
  {
    const auto& st = std::any_cast<const State&>(ck.state);
    u_ = st.u;
    left_bc_ = st.left_bc;
    right_bc_ = st.right_bc;
    time_ = ck.time;
  }

  InterfaceState GetInterface(std::string_view name) const override
  {
    auto it = produced_.find(std::string(name));
    if (it == produced_.end())
      throw std::runtime_error("HeatSubdomain '" + name_ +
                               "': no produced interface '" + std::string(name) +
                               "'");
    return InterfaceState(std::vector<Real>{SampleAt(it->second)});
  }

  void SetInterface(std::string_view name, const InterfaceState& s) override
  {
    auto it = consumed_.find(std::string(name));
    if (it == consumed_.end())
      throw std::runtime_error("HeatSubdomain '" + name_ +
                               "': no consumed interface '" + std::string(name) +
                               "'");
    const Real v = s[0];
    if (it->second == Side::Left)
      left_bc_ = v;
    else
      right_bc_ = v;
  }

  // Interior nodal field as the compared QoI for step doubling (plan Rung 1).
  std::span<const Real> ReportQoI() const override { return u_; }

  Capabilities GetCapabilities() const override
  {
    return Capabilities{/*can_restart=*/true, /*has_dense_output=*/false,
                        /*reports_qoi=*/true};
  }

  // --- inspection for the demo / reference ---------------------------------
  const std::vector<Real>& Coords() const noexcept { return x_; }
  const std::vector<Real>& Solution() const noexcept { return u_; }
  Real Time() const noexcept { return time_; }
  Real SampleAt(Real xs) const { return Interp(xs); }

private:
  struct State
  {
    std::vector<Real> u;
    Real left_bc;
    Real right_bc;
  };

  void Step(Real h)
  {
    switch (scheme_) {
      case TimeScheme::ForwardEuler: StepForwardEuler(h); break;
      case TimeScheme::BackwardEuler: StepBackwardEuler(h); break;
    }
  }

  void StepForwardEuler(Real h)
  {
    const std::size_t n = u_.size();
    const Real r = alpha_ * h / (dx_ * dx_);
    std::vector<Real> un(n);
    un.front() = left_bc_;
    un.back() = right_bc_;
    for (std::size_t i = 1; i + 1 < n; ++i)
      un[i] = u_[i] + r * (u_[i - 1] - 2.0 * u_[i] + u_[i + 1]);
    u_.swap(un);
  }

  // (I - r L) u^{n+1} = u^n with Dirichlet rows; solved by the Thomas algorithm.
  void StepBackwardEuler(Real h)
  {
    const std::size_t n = u_.size();
    const Real r = alpha_ * h / (dx_ * dx_);
    std::vector<Real> a(n, 0.0), b(n, 0.0), c(n, 0.0), d(n, 0.0);
    b.front() = 1.0;
    d.front() = left_bc_;
    b.back() = 1.0;
    d.back() = right_bc_;
    for (std::size_t i = 1; i + 1 < n; ++i) {
      a[i] = -r;
      b[i] = 1.0 + 2.0 * r;
      c[i] = -r;
      d[i] = u_[i];
    }
    // Forward elimination.
    for (std::size_t i = 1; i < n; ++i) {
      const Real m = a[i] / b[i - 1];
      b[i] -= m * c[i - 1];
      d[i] -= m * d[i - 1];
    }
    // Back substitution.
    u_.back() = d.back() / b.back();
    for (std::size_t i = n - 1; i-- > 0;)
      u_[i] = (d[i] - c[i] * u_[i + 1]) / b[i];
  }

  Real Interp(Real xs) const
  {
    const std::size_t n = x_.size();
    if (xs <= x_.front())
      return u_.front();
    if (xs >= x_.back())
      return u_.back();
    // Uniform grid ⇒ direct index; lands exactly on a node when xs is one.
    const Real fi = (xs - x_.front()) / dx_;
    auto i = static_cast<std::size_t>(std::floor(fi));
    if (i + 1 >= n)
      i = n - 2;
    const Real w = (xs - x_[i]) / dx_;
    return (1.0 - w) * u_[i] + w * u_[i + 1];
  }

  std::string name_;
  Real alpha_;
  Real internal_dt_;
  TimeScheme scheme_;
  Real dx_ = 0.0;
  Real time_ = 0.0;
  Real left_bc_ = 0.0;
  Real right_bc_ = 0.0;
  std::vector<Real> x_;
  std::vector<Real> u_;
  std::map<std::string, Real> produced_;
  std::map<std::string, Side> consumed_;
};

} // namespace pcms::transient::testing

#endif // PCMS_TRANSIENT_TESTING_HEAT1D_HPP
