#pragma once

#include "pcms/transient/participant.hpp"

#include <redev.h>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace pcms::transient
{

// Client-side runtime that exposes any Participant implementation to a remote
// transient driver through the participant protocol.
class ParticipantClient
{
public:
  struct Configuration
  {
    std::string channel;
    std::string produced_interface;
    std::string consumed_interface;
    std::size_t consumed_interface_size = 0;
    std::function<Real()> scalar_diagnostic;
  };

  struct FieldExchange
  {
    std::function<void()> send_produced_field;
    std::function<void()> receive_consumed_field;
  };

  ParticipantClient(redev::Redev& redev, Participant& participant,
                    Configuration configuration);

  void ConfigureFieldExchange(FieldExchange exchange);
  void Run();

private:
  void Send(const std::vector<double>& message);
  [[nodiscard]] std::vector<double> Receive();

  Participant* participant_;
  Configuration configuration_;
  redev::Channel channel_;
  redev::BidirectionalComm<double> communication_;
  std::size_t outbound_frame_size_ = 0;
  std::size_t inbound_frame_size_ = 0;
  FieldExchange field_exchange_;
};

} // namespace pcms::transient
