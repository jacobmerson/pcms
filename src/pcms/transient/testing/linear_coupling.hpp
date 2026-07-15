#ifndef PCMS_TRANSIENT_TESTING_LINEAR_COUPLING_HPP
#define PCMS_TRANSIENT_TESTING_LINEAR_COUPLING_HPP

#include "pcms/transient/participant.hpp"
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Linear-algebra test double (plan §9.1 LinearCouplingParticipant): AdvanceTo
// makes the produced interface a fixed affine function of the consumed one,
// state = gain·bc + source. Two of these coupled cyclically turn one Schwarz
// sweep into a linear fixed point x ← H x + b whose spectral radius is
// sqrt(gain0·gain1) — a tunable contraction factor ρ. This is the workhorse for
// accelerator / convergence / divergence tests: pure linear algebra, no physics
// (plan §9.3).
namespace pcms::transient::testing
{

class LinearParticipant final : public Participant
{
public:
  LinearParticipant(std::string name, std::string produced,
                    std::string consumed, Real gain, Real source)
    : name_(std::move(name)),
      produced_(std::move(produced)),
      consumed_(std::move(consumed)),
      gain_(gain),
      source_(source)
  {
  }

  std::string_view Name() const override { return name_; }

  void AdvanceTo(Real /*t_target*/) override { state_ = gain_ * bc_ + source_; }

  Checkpoint Save() const override
  {
    return Checkpoint{0.0, std::pair<Real, Real>{state_, bc_}};
  }
  void Restore(const Checkpoint& ck) override
  {
    const auto& s = std::any_cast<const std::pair<Real, Real>&>(ck.state);
    state_ = s.first;
    bc_ = s.second;
  }

  InterfaceState GetInterface(std::string_view name) const override
  {
    if (name != produced_)
      throw std::runtime_error("LinearParticipant: bad produced interface");
    return InterfaceState(std::vector<Real>{state_});
  }
  void SetInterface(std::string_view name, const InterfaceState& s) override
  {
    if (name != consumed_)
      throw std::runtime_error("LinearParticipant: bad consumed interface");
    bc_ = s[0];
  }

  Capabilities GetCapabilities() const override
  {
    return Capabilities{/*can_restart=*/true, false, false};
  }

private:
  std::string name_;
  std::string produced_;
  std::string consumed_;
  Real gain_;
  Real source_;
  Real state_ = 0.0;
  Real bc_ = 0.0;
};

} // namespace pcms::transient::testing

#endif // PCMS_TRANSIENT_TESTING_LINEAR_COUPLING_HPP
