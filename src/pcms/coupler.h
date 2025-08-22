#ifndef PCMS_COUPLER_H
#define PCMS_COUPLER_H
#include "pcms/common.h"
#include "pcms/field_communicator.h"
#include "pcms/field_adapter_interface.h"
#include "pcms/adapter/omega_h/omega_h_field.h"
#include "pcms/profile.h"

namespace pcms
{

class CoupledField
{
public:
  template <typename FieldAdapterT>
  CoupledField(const std::string& name, FieldAdapterT field_adapter,
               MPI_Comm mpi_comm, redev::Redev& redev, redev::Channel& channel,
               bool participates = true)
    : field_adapter_(std::unique_ptr<IFieldAdapter>(new FieldAdapterT(std::move(field_adapter))))
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
    // Create field communicator - this will need to be adapted based on the actual FieldCommunicator interface
    // For now, we'll keep this simplified
    mpi_comm_subset_ = mpi_comm_subset;
    participates_ = participates;
  }

  void Send(Mode mode = Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    // Implementation will depend on how FieldCommunicator is refactored
    // For now, this is a placeholder
  }
  void Receive(Mode mode = Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    // Implementation will depend on how FieldCommunicator is refactored
    // For now, this is a placeholder
  }
  // Type-safe accessor with better error handling
  template <typename T>
  [[nodiscard]] T* GetFieldAdapter() {
    PCMS_FUNCTION_TIMER;
    return dynamic_cast<T*>(field_adapter_.get());
  }
  
  template <typename T>
  [[nodiscard]] const T* GetFieldAdapter() const {
    PCMS_FUNCTION_TIMER;
    return dynamic_cast<const T*>(field_adapter_.get());
  }
  
  // Get adapter by interface
  [[nodiscard]] IFieldAdapter* GetFieldAdapter() {
    return field_adapter_.get();
  }
  
  [[nodiscard]] const IFieldAdapter* GetFieldAdapter() const {
    return field_adapter_.get();
  }
  
  // Get adapter type
  [[nodiscard]] FieldAdapterType GetAdapterType() const {
    return field_adapter_->GetAdapterType();
  }
  // Destructor to clean up MPI communicator
  ~CoupledField() {
    PCMS_FUNCTION_TIMER;
    if (mpi_comm_subset_ != MPI_COMM_NULL) {
      MPI_Comm_free(&mpi_comm_subset_);
    }
  }
  
  // Copy and move semantics
  CoupledField(const CoupledField&) = delete;
  CoupledField& operator=(const CoupledField&) = delete;
  CoupledField(CoupledField&&) = default;
  CoupledField& operator=(CoupledField&&) = default;

private:
  std::unique_ptr<IFieldAdapter> field_adapter_;
  MPI_Comm mpi_comm_subset_ = MPI_COMM_NULL;
  bool participates_ = true;
  // Note: FieldCommunicator will need to be refactored to work with the new interface
  // This is a simplified version for now
};

class Application
{
public:
  Application(std::string name, MPI_Comm comm,
              redev::Redev& redev, adios2::Params params,
              redev::TransportType transport_type,
              std::string path)
    : mpi_comm_(comm),
      redev_(redev),
      channel_{redev_.CreateAdiosChannel(std::move(name), std::move(params),
                                      transport_type, std::move(path))}
  {
    PCMS_FUNCTION_TIMER;
  }
  // FIXME should take a file path for the parameters, not take adios2 params.
  // These fields are supposed to be agnostic to adios2...
  template <typename FieldAdapterT>
  CoupledField* AddField(
    std::string name, FieldAdapterT&& field_adapter, bool participates = true)
  {
    PCMS_FUNCTION_TIMER;
    auto [it, inserted] = fields_.template try_emplace(
      name, name, std::forward<FieldAdapterT>(field_adapter), mpi_comm_, redev_,
      channel_, participates);
    if (!inserted) {
      std::cerr << "Field with this name" << name << "already exists!\n";
      std::terminate();
    }
    return &(it->second);
  }
  void SendField(const std::string& name, Mode mode = Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    PCMS_ALWAYS_ASSERT(InSendPhase());
    detail::find_or_error(name, fields_).Send(mode);
  };
  void ReceiveField(const std::string& name, Mode mode = Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    PCMS_ALWAYS_ASSERT(InReceivePhase());
    detail::find_or_error(name, fields_).Receive(mode);
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

  template <typename Func, typename... Args>
  auto SendPhase(const Func& func, Args&&... args)
  {
    PCMS_FUNCTION_TIMER;
    return channel_.SendPhase(func, std::forward<Args>(args)...);
  }
  template <typename Func, typename... Args>
  auto ReceivePhase(const Func& func, Args&&... args)
  {
    PCMS_FUNCTION_TIMER;
    return channel_.ReceivePhase(func, std::forward<Args>(args)...);
  }

private:
  MPI_Comm mpi_comm_;
  redev::Redev& redev_;
  redev::Channel channel_;
  // map is used rather than unordered_map because we give pointers to the
  // internal data and rehash of unordered_map can cause pointer invalidation.
  // map is less cache friendly, but pointers are not invalidated.
  std::map<std::string, CoupledField> fields_;
};

class Coupler
{
private:
  redev::Redev SetUpRedev(bool isServer, redev::Partition partition) {
    if (isServer)
      return redev::Redev(mpi_comm_, std::move(partition), ProcessType::Server);
    else
      return redev::Redev(mpi_comm_);
  }
public:
  Coupler(std::string name, MPI_Comm comm, bool isServer, redev::Partition partition)
    : name_(std::move(name)),
      mpi_comm_(comm),
      redev_(SetUpRedev(isServer, std::move(partition)))
  {
    PCMS_FUNCTION_TIMER;
  }
  Application* AddApplication(
    std::string name, std::string path = "",
    redev::TransportType transport_type = redev::TransportType::BP4,
    adios2::Params params = {{"Streaming", "On"}, {"OpenTimeoutSecs", "60"}})
  {
    PCMS_FUNCTION_TIMER;
    auto key = path + name;
    auto [it, inserted] = applications_.template try_emplace(
      key, std::move(name), mpi_comm_, redev_,
      std::move(params), transport_type, std::move(path));
    if (!inserted) {
      std::cerr << "Application with name " << name << "already exists!\n";
      std::terminate();
    }
    return &(it->second);
  }

  [[nodiscard]] const redev::Partition& GetPartition() const noexcept
  {
    return redev_.GetPartition();
  }

private:
  std::string name_;
  MPI_Comm mpi_comm_;
  redev::Redev redev_;
  // gather and scatter operations have reference to internal fields
  std::map<std::string, Application> applications_;
};

} // namespace pcms

#endif // PCMS_COUPLER_H
