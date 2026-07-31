#pragma once

#include "pcms/transient/participant.hpp"

#include <redev.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace pcms::transient
{

// Server-side Participant proxy. Every operation is executed by an
// external participant process through its own Redev channel.
class RemoteParticipant final : public Participant
{
public:
  struct Configuration
  {
    std::string channel;
    std::string participant;
    std::string produced_interface;
    std::string consumed_interface;
    // Multi-client Redev servers must create every channel before any channel
    // begins communication. Leave false for the existing eager behavior.
    bool defer_connect = false;
  };

  struct FieldExchange
  {
    std::function<void()> receive_produced_field;
    std::function<void(const InterfaceState&)> send_consumed_field;
  };

  RemoteParticipant(redev::Redev& redev, Configuration configuration);
  void ConfigureFieldExchange(FieldExchange exchange);
  void Connect();

  [[nodiscard]] std::string_view Name() const override;
  void AdvanceTo(Real target_time) override;
  [[nodiscard]] Checkpoint Save() const override;
  void Restore(const Checkpoint& checkpoint) override;
  [[nodiscard]] InterfaceState GetInterface(
    std::string_view name) const override;
  void SetInterface(
    std::string_view name,
    const InterfaceState& state) override;
  [[nodiscard]] Capabilities GetCapabilities() const override;

  [[nodiscard]] std::size_t InterfaceSize() const noexcept;
  // Optional application-defined scalar diagnostic. Its meaning is agreed by
  // the participant-adapter configuration rather than prescribed by the
  // protocol.
  [[nodiscard]] double QueryScalar() const;
  void Shutdown();

private:
  struct RemoteCheckpoint
  {
    std::int64_t token;
  };

  void Send(const std::vector<double>& message) const;
  [[nodiscard]] std::vector<double> Receive() const;
  [[nodiscard]] std::vector<double> Request(
    const std::vector<double>& message) const;
  static void ExpectOk(const std::vector<double>& response);

  std::string participant_name_;
  std::string produced_interface_;
  std::string consumed_interface_;
  mutable redev::Channel channel_;
  mutable redev::BidirectionalComm<double> communication_;
  std::size_t outbound_frame_size_ = 0;
  std::size_t interface_size_ = 0;
  Capabilities capabilities_;
  Real current_time_ = 0.0;
  mutable std::int64_t next_checkpoint_token_ = 0;
  bool shutdown_ = false;
  bool connected_ = false;
  FieldExchange field_exchange_;
};

} // namespace pcms::transient
