#ifndef FIELD_LAYOUT_COMMUNICATOR_H_
#define FIELD_LAYOUT_COMMUNICATOR_H_

#include "pcms/field_layout.h"
#include "pcms/field.h"
#include "pcms/profile.h"
#include "pcms/assert.h"
#include "pcms/arrays.h"
#include <redev.h>

namespace pcms
{

template <typename T>
class FieldLayoutCommunicator
{
public:
  FieldLayoutCommunicator(std::string name, MPI_Comm mpi_comm,
                          redev::Redev& redev, redev::Channel& channel,
                          FieldLayout& layout)
    : mpi_comm_(mpi_comm),
      channel_(channel),
      layout_(layout),
      name_{std::move(name)},
      redev_(redev),
      msg_size_(0)
  {
    comm_ = channel.CreateComm<T>(name_, mpi_comm_);
    gid_comm_ = channel.CreateComm<GO>(name_ + "_gids", mpi_comm_);
    if (mpi_comm != MPI_COMM_NULL) {
      UpdateLayout();
    } else {
      UpdateLayoutNull();
    }
  }

  void Send(T* msg, redev::Mode mode = redev::Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    comm_.Send(msg, mode);
  }

  std::vector<T> Recv(redev::Mode mode = redev::Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    return comm_.Recv(mode);
  }

  Rank1View<const pcms::LO, pcms::HostMemorySpace> GetPermutationArray() const
  {
    return make_const_array_view(message_permutation_);
  }

  const redev::Channel& GetChannel() const { return channel_; }

  const FieldLayout& GetLayout() const { return layout_; }

  size_t GetMsgSize() const { return msg_size_; }

  void UpdateLayout()
  {
    PCMS_FUNCTION_TIMER;
    if (mpi_comm_ == MPI_COMM_NULL) {
      UpdateLayoutNull();
      return;
    }
    if (redev_.GetProcessType() == redev::ProcessType::Client) {
      auto plan =
        layout_.BuildClientPlan(redev::Partition{redev_.GetPartition()});
      ApplyClientPlan(std::move(plan));
    } else {
      channel_.BeginReceiveCommunicationPhase();
      auto recv_gids = gid_comm_.Recv();
      channel_.EndReceiveCommunicationPhase();
      int rank = 0;
      int nproc = 1;
      MPI_Comm_rank(mpi_comm_, &rank);
      MPI_Comm_size(mpi_comm_, &nproc);
      GlobalIDView<HostMemorySpace> recv_gids_view(recv_gids.data(),
                                                   recv_gids.size());
      auto plan = layout_.BuildServerPlan(
        recv_gids_view, gid_comm_.GetInMessageLayout(), rank, nproc);
      ApplyServerPlan(std::move(plan));
      msg_size_ = recv_gids.size();
    }
  }

  void UpdateLayoutNull()
  {
    PCMS_FUNCTION_TIMER;
    if (redev_.GetProcessType() == redev::ProcessType::Client) {
      channel_.BeginSendCommunicationPhase();
      channel_.EndSendCommunicationPhase();
    } else {
      channel_.BeginReceiveCommunicationPhase();
      channel_.EndReceiveCommunicationPhase();
    }
  }

private:
  void ApplyClientPlan(FieldLayoutPlan&& plan)
  {
    message_permutation_ = std::move(plan.permutation);
    msg_size_ = plan.gid_payload.size();
    comm_.SetOutMessageLayout(plan.destinations, plan.offsets);
    gid_comm_.SetOutMessageLayout(plan.destinations, plan.offsets);
    channel_.BeginSendCommunicationPhase();
    gid_comm_.Send(plan.gid_payload.data());
    channel_.EndSendCommunicationPhase();
  }

  void ApplyServerPlan(FieldLayoutPlan&& plan)
  {
    message_permutation_ = std::move(plan.permutation);
    comm_.SetOutMessageLayout(plan.destinations, plan.offsets);
  }

  MPI_Comm mpi_comm_;
  redev::Channel& channel_;
  std::vector<pcms::LO> message_permutation_;
  redev::BidirectionalComm<T> comm_;
  redev::BidirectionalComm<GO> gid_comm_;
  FieldLayout& layout_;
  redev::Redev& redev_;
  std::string name_;
  size_t msg_size_;
};
} // namespace pcms

#endif // FIELD_LAYOUT_COMMUNICATOR_H_
