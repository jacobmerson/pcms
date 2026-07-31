#include "pcms/transient/participant_adapter/participant_client.hpp"

#include "pcms/transient/participant_adapter/participant_protocol.hpp"

#include <adios2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>

namespace pcms::transient
{
namespace
{

using protocol::Command;

double Encode(Command command)
{
  return static_cast<double>(command);
}

Command Decode(double value)
{
  return static_cast<Command>(static_cast<int>(std::llround(value)));
}

} // namespace

ParticipantClient::ParticipantClient(
  redev::Redev& redev, Participant& participant,
  Configuration configuration)
  : participant_(&participant),
    configuration_(std::move(configuration)),
    channel_(redev.CreateAdiosChannel(
      configuration_.channel,
      adios2::Params{{"Streaming", "On"}, {"OpenTimeoutSecs", "60"}},
      redev::TransportType::SST)),
    communication_(channel_.CreateComm<double>(
      configuration_.channel, redev.GetMPIComm()))
{
  if (configuration_.channel.empty() ||
      configuration_.produced_interface.empty() ||
      configuration_.consumed_interface.empty() ||
      configuration_.consumed_interface_size == 0) {
    throw std::invalid_argument(
      "ParticipantClient: incomplete configuration");
  }
  outbound_frame_size_ = std::max(
    protocol::HelloSize,
    protocol::HeaderSize +
      participant_->GetInterface(configuration_.produced_interface).Size());
  inbound_frame_size_ =
    protocol::HeaderSize + configuration_.consumed_interface_size;
}

void ParticipantClient::ConfigureFieldExchange(FieldExchange exchange)
{
  if (!exchange.send_produced_field ||
      !exchange.receive_consumed_field) {
    throw std::invalid_argument(
      "ParticipantClient: incomplete PCMS field exchange");
  }
  field_exchange_ = std::move(exchange);
}

void ParticipantClient::Run()
{
  if (!field_exchange_.send_produced_field ||
      !field_exchange_.receive_consumed_field) {
    throw std::runtime_error(
      "ParticipantClient: PCMS field exchange is not configured");
  }
  const Capabilities capabilities = participant_->GetCapabilities();
  Send({Encode(Command::hello),
        static_cast<double>(configuration_.consumed_interface_size),
        capabilities.can_restart ? 1.0 : 0.0,
        capabilities.has_dense_output ? 1.0 : 0.0,
        capabilities.reports_qoi ? 1.0 : 0.0});
  const auto hello_response = Receive();
  if (hello_response.empty() ||
      Decode(hello_response[0]) != Command::ok) {
    throw std::runtime_error(
      "ParticipantClient: server rejected handshake");
  }

  std::map<std::int64_t, Checkpoint> checkpoints;
  bool running = true;
  while (running) {
    const auto request = Receive();
    if (request.size() < protocol::HeaderSize)
      throw std::runtime_error(
        "ParticipantClient: malformed request");

    switch (Decode(request[0])) {
    case Command::save: {
      const auto token =
        static_cast<std::int64_t>(std::llround(request[1]));
      checkpoints.insert_or_assign(token, participant_->Save());
      Send({Encode(Command::ok), 0.0});
      break;
    }
    case Command::restore: {
      const auto token =
        static_cast<std::int64_t>(std::llround(request[1]));
      const auto checkpoint = checkpoints.find(token);
      if (checkpoint == checkpoints.end())
        throw std::runtime_error(
          "ParticipantClient: unknown checkpoint token");
      participant_->Restore(checkpoint->second);
      Send({Encode(Command::ok), 0.0});
      break;
    }
    case Command::advance:
      participant_->AdvanceTo(request[1]);
      Send({Encode(Command::ok), 0.0});
      break;
    case Command::get_field: {
      field_exchange_.send_produced_field();
      Send({Encode(Command::ok), 0.0});
      break;
    }
    case Command::set_boundary: {
      field_exchange_.receive_consumed_field();
      participant_->SetInterface(
        configuration_.consumed_interface,
        InterfaceState{});
      Send({Encode(Command::ok), 0.0});
      break;
    }
    case Command::get_scalar:
      if (!configuration_.scalar_diagnostic)
        throw std::runtime_error(
          "ParticipantClient: no scalar diagnostic configured");
      Send({Encode(Command::ok),
            configuration_.scalar_diagnostic()});
      break;
    case Command::shutdown:
      Send({Encode(Command::ok), 0.0});
      running = false;
      break;
    default:
      throw std::runtime_error(
        "ParticipantClient: unknown command");
    }
  }
}

void ParticipantClient::Send(const std::vector<double>& message)
{
  if (message.empty())
    throw std::invalid_argument(
      "ParticipantClient: cannot send an empty message");
  if (message.size() > outbound_frame_size_)
    throw std::runtime_error(
      "ParticipantClient: message exceeds fixed transport frame");
  std::vector<double> frame(outbound_frame_size_, 0.0);
  std::copy(message.begin(), message.end(), frame.begin());
  redev::LOs destinations{0};
  redev::LOs offsets{
    0, static_cast<redev::LO>(frame.size())};
  communication_.SetOutMessageLayout(destinations, offsets);
  channel_.BeginSendCommunicationPhase();
  communication_.Send(frame.data());
  channel_.EndSendCommunicationPhase();
}

std::vector<double> ParticipantClient::Receive()
{
  channel_.BeginReceiveCommunicationPhase();
  auto message = communication_.Recv();
  channel_.EndReceiveCommunicationPhase();
  if (message.size() != inbound_frame_size_)
    throw std::runtime_error(
      "ParticipantClient: invalid fixed transport frame");
  return message;
}

} // namespace pcms::transient
