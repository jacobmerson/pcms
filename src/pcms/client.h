#ifndef PCMS_COUPLING_CLIENT_H
#define PCMS_COUPLING_CLIENT_H
#include "pcms/common.h"
#include "pcms/field_communicator.h"
#include "pcms/profile.h"


namespace pcms
{

class CoupledField
{
public:
  template <typename FieldAdapterT>
  CoupledField(const std::string& name, FieldAdapterT field_adapter,
               MPI_Comm mpi_comm, redev::Redev& redev, redev::Channel& channel,
               bool participates)
  {
    PCMS_FUNCTION_TIMER;
    MPI_Comm mpi_comm_subset = MPI_COMM_NULL;
    PCMS_ALWAYS_ASSERT((mpi_comm == MPI_COMM_NULL) ? (participates == false)
                                                     : true);
    if (mpi_comm != MPI_COMM_NULL) {
      int rank = -1;
      MPI_Comm_rank(mpi_comm, &rank);
      MPI_Comm_split(mpi_comm, participates ? 0 : MPI_UNDEFINED, rank,
                     &mpi_comm_subset);
    }
    coupled_field_ =
      std::make_unique<CoupledFieldModel<FieldAdapterT, FieldAdapterT>>(
        name, std::move(field_adapter), mpi_comm_subset, redev, channel,
        participates);
  }

  void Send(Mode mode = Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    coupled_field_->Send(mode);
  }
  void Receive()
  {
    PCMS_FUNCTION_TIMER;
    coupled_field_->Receive();
  }
  struct CoupledFieldConcept
  {
    virtual void Send(Mode) = 0;
    virtual void Receive() = 0;
    virtual ~CoupledFieldConcept() = default;
  };
  template <typename FieldAdapterT, typename CommT>
  struct CoupledFieldModel final : CoupledFieldConcept
  {
    using value_type = typename FieldAdapterT::value_type;

    CoupledFieldModel(const std::string& name, FieldAdapterT&& field_adapter,
                      MPI_Comm mpi_comm_subset, redev::Redev& redev,
                      redev::Channel& channel, bool participates)
      : mpi_comm_subset_(mpi_comm_subset),
        field_adapter_(std::move(field_adapter)),
        comm_(FieldCommunicator<CommT>(name, mpi_comm_subset_, redev, channel,
                                       field_adapter_))
    {
      PCMS_FUNCTION_TIMER;
    }
    void Send(Mode mode) final
    {
      PCMS_FUNCTION_TIMER;
      comm_.Send(mode);
    };
    void Receive() final
    {
      PCMS_FUNCTION_TIMER;
      comm_.Receive();
    };
    ~CoupledFieldModel()
    {
      PCMS_FUNCTION_TIMER;
      if (mpi_comm_subset_ != MPI_COMM_NULL)
        MPI_Comm_free(&mpi_comm_subset_);
    }

    MPI_Comm mpi_comm_subset_;
    FieldAdapterT field_adapter_;
    FieldCommunicator<CommT> comm_;
  };

private:
  std::unique_ptr<CoupledFieldConcept> coupled_field_;
};



class CouplerClient
{
public:
  CouplerClient(std::string name, MPI_Comm comm,
                redev::TransportType transport_type = redev::TransportType::BP4,
                adios2::Params params = {{"Streaming", "On"},
                                         {"OpenTimeoutSecs", "400"}},
                std::string path = "")
    : name_(std::move(name)),
      mpi_comm_(comm),
      redev_(comm),
      channel_{redev_.CreateAdiosChannel(name_, std::move(params),
                                         transport_type, std::move(path))}
  {
    PCMS_FUNCTION_TIMER;
  }

  [[nodiscard]] const redev::Partition& GetPartition() const
  {
    PCMS_FUNCTION_TIMER;
    return redev_.GetPartition();
  }

  /**
   * @known_issue
   * The redev partion has to be same as the partition of the OH mesh.
   * It asserts the number of elements sent and received are same.
   * otherwise, ConstructPermutation() will fail.
  */
  template <typename FieldAdapterT>
  CoupledField* AddField(std::string name, FieldAdapterT field_adapter,
                         bool participates = true)
  {
    PCMS_FUNCTION_TIMER;
    auto [it, inserted] =
      fields_.template try_emplace(name, name, std::move(field_adapter),
                                   mpi_comm_, redev_, channel_, participates);
    if (!inserted) {
      std::cerr << "OHField with this name" << name << "already exists!\n";
      std::terminate();
    }
    return &(it->second);
  }

  // take a string& since map cannot be searched with string_view
  // (heterogeneous lookup)
  void SendField(const std::string& name, Mode mode = Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    PCMS_ALWAYS_ASSERT(InSendPhase());
    detail::find_or_error(name, fields_).Send(mode);
  };
  // take a string& since map cannot be searched with string_view
  // (heterogeneous lookup)
  void ReceiveField(const std::string& name)
  {
    PCMS_FUNCTION_TIMER;
    PCMS_ALWAYS_ASSERT(InReceivePhase());
    detail::find_or_error(name, fields_).Receive();
  };
  [[nodiscard]] bool InSendPhase() const noexcept
  {
    PCMS_FUNCTION_TIMER;
    return channel_.InSendCommunicationPhase();
  }
  [[nodiscard]] bool InReceivePhase() const noexcept
  {
    PCMS_FUNCTION_TIMER;
    return channel_.InReceiveCommunicationPhase();
  }
  void BeginSendPhase()
  {
    PCMS_FUNCTION_TIMER;
    channel_.BeginSendCommunicationPhase();
  }
  void EndSendPhase()
  {
    PCMS_FUNCTION_TIMER;
    channel_.EndSendCommunicationPhase();
  }
  void BeginReceivePhase()
  {
    PCMS_FUNCTION_TIMER;
    channel_.BeginReceiveCommunicationPhase();
  }
  void EndReceivePhase()
  {
    PCMS_FUNCTION_TIMER;
    channel_.EndReceiveCommunicationPhase();
  }

private:
  std::string name_;
  MPI_Comm mpi_comm_;
  redev::Redev redev_;
  // map rather than unordered_map is necessary to avoid iterator invalidation.
  // This is important because we pass pointers to the fields out of this class
  std::map<std::string, CoupledField> fields_;
  redev::Channel channel_;
};
} // namespace pcms

#endif // PCMS_COUPLING_CLIENT_H