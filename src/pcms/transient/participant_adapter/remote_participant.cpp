#include "pcms/transient/participant_adapter/remote_participant.hpp"

#include "pcms/transient/participant_adapter/participant_protocol.hpp"

#include <adios2.h>

#include <algorithm>
#include <cmath>
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

RemoteParticipant::RemoteParticipant(
  redev::Redev& redev, Configuration configuration)
  : participant_name_(std::move(configuration.participant)),
    produced_interface_(std::move(configuration.produced_interface)),
    consumed_interface_(std::move(configuration.consumed_interface)),
    channel_(redev.CreateAdiosChannel(
      configuration.channel,
      adios2::Params{{"Streaming", "On"}, {"OpenTimeoutSecs", "60"}},
      redev::TransportType::SST)),
    communication_(
      channel_.CreateComm<double>(configuration.channel, redev.GetMPIComm()))
{
  if (!configuration.defer_connect)
    Connect();
}

void RemoteParticipant::ConfigureFieldExchange(FieldExchange exchange)
{
  if (!exchange.receive_produced_field ||
      !exchange.send_consumed_field) {
    throw std::invalid_argument(
      "RemoteParticipant: incomplete PCMS field exchange");
  }
  field_exchange_ = std::move(exchange);
}

void RemoteParticipant::Connect()
{
  if (connected_)
    return;

  const auto hello = Receive();
  if (hello.size() < protocol::HelloSize ||
      Decode(hello[0]) != Command::hello || hello[1] <= 0.0) {
    throw std::runtime_error(
      "RemoteParticipant: invalid participant handshake");
  }
  interface_size_ =
    static_cast<std::size_t>(std::llround(hello[1]));
  outbound_frame_size_ = protocol::HeaderSize + interface_size_;
  capabilities_ = Capabilities{
    /*can_restart=*/hello[2] != 0.0,
    /*has_dense_output=*/hello[3] != 0.0,
    /*reports_qoi=*/hello[4] != 0.0};
  Send({Encode(Command::ok), 0.0});
  connected_ = true;
}

std::string_view RemoteParticipant::Name() const
{
  return participant_name_;
}

void RemoteParticipant::AdvanceTo(Real target_time)
{
  ExpectOk(Request({Encode(Command::advance), target_time}));
  current_time_ = target_time;
}

Checkpoint RemoteParticipant::Save() const
{
  const std::int64_t token = ++next_checkpoint_token_;
  ExpectOk(Request(
    {Encode(Command::save), static_cast<double>(token)}));
  return Checkpoint{current_time_, RemoteCheckpoint{token}};
}

void RemoteParticipant::Restore(const Checkpoint& checkpoint)
{
  const auto& remote = std::any_cast<const RemoteCheckpoint&>(
    checkpoint.state);
  ExpectOk(Request(
    {Encode(Command::restore), static_cast<double>(remote.token)}));
  current_time_ = checkpoint.time;
}

InterfaceState RemoteParticipant::GetInterface(
  std::string_view name) const
{
  if (name != produced_interface_)
    throw std::invalid_argument(
      "RemoteParticipant: unknown produced interface");

  if (!field_exchange_.receive_produced_field)
    throw std::runtime_error(
      "RemoteParticipant: PCMS field receive is not configured");
  Send({Encode(Command::get_field), 0.0});
  field_exchange_.receive_produced_field();
  ExpectOk(Receive());
  // The distributed field now resides in the PCMS coupler-side Field bound
  // by field_exchange_. The transient vector is produced by the coupling
  // backend only after PCMS interpolation.
  return InterfaceState{};
}

void RemoteParticipant::SetInterface(
  std::string_view name, const InterfaceState& state)
{
  if (name != consumed_interface_)
    throw std::invalid_argument(
      "RemoteParticipant: unknown consumed interface");
  if (state.Size() != interface_size_)
    throw std::invalid_argument(
      "RemoteParticipant: interface size mismatch");

  if (!field_exchange_.send_consumed_field)
    throw std::runtime_error(
      "RemoteParticipant: PCMS field send is not configured");
  Send({Encode(Command::set_boundary),
        static_cast<double>(state.Size())});
  field_exchange_.send_consumed_field(state);
  ExpectOk(Receive());
}

Capabilities RemoteParticipant::GetCapabilities() const
{
  return capabilities_;
}

std::size_t RemoteParticipant::InterfaceSize() const noexcept
{
  return interface_size_;
}

double RemoteParticipant::QueryScalar() const
{
  const auto response =
    Request({Encode(Command::get_scalar), 0.0});
  ExpectOk(response);
  if (response.size() < protocol::HeaderSize)
    throw std::runtime_error(
      "RemoteParticipant: invalid scalar response");
  return response[1];
}

void RemoteParticipant::Shutdown()
{
  if (shutdown_)
    return;
  ExpectOk(Request({Encode(Command::shutdown), 0.0}));
  shutdown_ = true;
}

void RemoteParticipant::Send(
  const std::vector<double>& message) const
{
  if (message.empty())
    throw std::invalid_argument(
      "RemoteParticipant: cannot send an empty message");
  if (message.size() > outbound_frame_size_)
    throw std::runtime_error(
      "RemoteParticipant: message exceeds fixed transport frame");
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

std::vector<double> RemoteParticipant::Receive() const
{
  channel_.BeginReceiveCommunicationPhase();
  auto message = communication_.Recv();
  channel_.EndReceiveCommunicationPhase();
  return message;
}

std::vector<double> RemoteParticipant::Request(
  const std::vector<double>& message) const
{
  Send(message);
  return Receive();
}

void RemoteParticipant::ExpectOk(
  const std::vector<double>& response)
{
  if (response.empty() || Decode(response[0]) != Command::ok)
    throw std::runtime_error(
      "RemoteParticipant: participant operation failed");
}

} // namespace pcms::transient
