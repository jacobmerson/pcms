#ifndef COUPLER2_H_
#define COUPLER2_H_

#include "pcms/coupler/coupler_types.h"
#include "pcms/field/field.h"
#include "pcms/field/field_layout.h"
#include "pcms/coupler/field_layout_communicator.h"
#include "pcms/coupler/field_communicator.hpp"
#include "pcms/coupler/field_exchange_planner.h"
#include "pcms/coupler/overlap_mask.h"
#include "pcms/transfer/transfer_operator.hpp"
#include "pcms/utility/assert.h"
#include "pcms/utility/common.h"
#include "pcms/utility/profile.h"
#include <memory>

namespace pcms
{

class Application;
class Coupler;

template <typename T>
class FieldHandle
{
public:
  FieldHandle(Application* app, std::string name)
    : app_(app), name_(std::move(name))
  {
  }

  void Send(redev::Mode mode = redev::Mode::Synchronous) const;
  void Receive(redev::Mode mode = redev::Mode::Synchronous) const;
  [[nodiscard]] Field<T>& GetField() const;

private:
  Application* app_;
  std::string name_;
};

class Transfer
{
public:
  virtual void Run() const = 0;
  virtual ~Transfer() noexcept = default;
};

template <typename T>
class BoundTransfer final : public Transfer
{
public:
  BoundTransfer(FieldHandle<T> source, FieldHandle<T> target,
                std::unique_ptr<TransferOperator<T>> transfer)
    : source_(std::move(source)),
      target_(std::move(target)),
      transfer_(std::move(transfer))
  {
    PCMS_ALWAYS_ASSERT(transfer_ != nullptr);
  }

  void Run() const override
  {
    PCMS_FUNCTION_TIMER;
    transfer_->Apply(source_.GetField(), target_.GetField());
  }

private:
  FieldHandle<T> source_;
  FieldHandle<T> target_;
  std::unique_ptr<TransferOperator<T>> transfer_;
};

class TransferHandle
{
public:
  void Run() const;

private:
  TransferHandle(Coupler* coupler, std::size_t id)
    : coupler_(coupler), id_(id)
  {
  }

  friend class Coupler;

  Coupler* coupler_;
  std::size_t id_;
};

class Application
{
public:
  Application(std::string name, MPI_Comm comm, redev::Redev& redev,
              adios2::Params params, redev::TransportType transport_type,
              std::string path)
    : mpi_comm_(comm),
      redev_(redev),
      channel_{redev_.CreateAdiosChannel(std::move(name), std::move(params),
                                         transport_type, std::move(path))}
  {
    PCMS_FUNCTION_TIMER;
  }

  const FieldLayout& AddLayout(std::string name,
                               std::shared_ptr<const FieldLayout> layout,
                               bool participates = true);
  const FieldLayout& AddLayout(std::string name,
                               std::shared_ptr<const FieldLayout> layout,
                               std::unique_ptr<FieldExchangePlanner> planner,
                               bool participates = true);

  // Set the overlap mask for a specific layout by name
  void SetLayoutOverlapMask(const std::string& layout_name,
                            std::unique_ptr<OverlapMask> overlap_mask);

  template <typename T>
  FieldHandle<T> AddField(std::string name, Field<T>&& field,
                          bool participates = true);

  template <typename T>
  FieldHandle<T> AddField(std::string name, Field<T>&& field,
                          std::unique_ptr<FieldSerializer<T>> serializer,
                          bool participates = true);

  void SendField(const std::string& name,
                 redev::Mode mode = redev::Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    PCMS_ALWAYS_ASSERT(InSendPhase());
    FieldCommunicator2Ptr& communicator =
      detail::find_or_error(name, field_communicators_);
    std::visit(
      [mode](auto& field_communicator) { field_communicator->Send(mode); },
      communicator);
  };
  void ReceiveField(const std::string& name,
                    redev::Mode mode = redev::Mode::Synchronous)
  {
    PCMS_FUNCTION_TIMER;
    PCMS_ALWAYS_ASSERT(InReceivePhase());
    std::visit(
      [mode](auto& field_communicator) { field_communicator->Receive(); },
      detail::find_or_error(name, field_communicators_));
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

  [[nodiscard]] std::size_t GetLayoutCommunicatorCount() const noexcept
  {
    return field_layout_communicators_.size();
  }

  template <typename T>
  [[nodiscard]] Field<T>& GetField(const std::string& name);

private:
  FieldLayoutCommunicator& GetLayoutCommunicator(const FieldLayout& layout);

  MPI_Comm mpi_comm_;
  redev::Redev& redev_;
  redev::Channel channel_;
  std::vector<std::shared_ptr<const FieldLayout>> layouts_;
  std::map<std::string, FieldVariant> fields_;
  std::vector<std::unique_ptr<FieldLayoutCommunicator>>
    owned_field_layout_communicators_;
  // map is used rather than unordered_map because we give pointers to the
  // internal data and rehash of unordered_map can cause pointer invalidation.
  // map is less cache friendly, but pointers are not invalidated.
  std::map<std::string, FieldCommunicator2Ptr> field_communicators_;
  std::map<const FieldLayout*, std::unique_ptr<FieldLayoutCommunicator>>
    field_layout_communicators_;
  std::map<std::string, std::unique_ptr<OverlapMask>> layout_overlap_masks_;
};

class Coupler
{
private:
  redev::Redev SetUpRedev(bool isServer, redev::Partition partition)
  {
    if (isServer)
      return redev::Redev(mpi_comm_, std::move(partition), ProcessType::Server);
    else
      return redev::Redev(mpi_comm_);
  }

public:
  Coupler(std::string name, MPI_Comm comm, bool isServer,
          redev::Partition partition)
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
    auto [it, inserted] = applications_.try_emplace(
      key, std::move(name), mpi_comm_, redev_, std::move(params),
      transport_type, std::move(path));
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

  template <typename T>
  TransferHandle AddTransfer(
    FieldHandle<T> source, FieldHandle<T> target,
    std::unique_ptr<TransferOperator<T>> transfer);

  void RunTransfer(std::size_t id) { transfers_.at(id)->Run(); }
private:
  std::string name_;
  MPI_Comm mpi_comm_;
  redev::Redev redev_;
  // gather and scatter operations have reference to internal fields
  std::map<std::string, Application> applications_;
  std::vector<std::unique_ptr<Transfer>> transfers_;

};

} // namespace pcms

template <typename T>
void pcms::FieldHandle<T>::Send(redev::Mode mode) const
{
  PCMS_ALWAYS_ASSERT(app_ != nullptr);
  app_->SendField(name_, mode);
}

template <typename T>
void pcms::FieldHandle<T>::Receive(redev::Mode mode) const
{
  PCMS_ALWAYS_ASSERT(app_ != nullptr);
  app_->ReceiveField(name_, mode);
}

template <typename T>
pcms::Field<T>& pcms::FieldHandle<T>::GetField() const
{
  PCMS_ALWAYS_ASSERT(app_ != nullptr);
  return app_->GetField<T>(name_);
}

template <typename T>
pcms::Field<T>& pcms::Application::GetField(const std::string& name)
{
  auto* field = std::get_if<Field<T>>(&detail::find_or_error(name, fields_));
  if (field == nullptr) {
    throw pcms_error("Field stored with different type than requested");
  }
  return *field;
}

inline void pcms::TransferHandle::Run() const
{
  PCMS_ALWAYS_ASSERT(coupler_ != nullptr);
  coupler_->RunTransfer(id_);
}

template <typename T>
pcms::TransferHandle pcms::Coupler::AddTransfer(
  FieldHandle<T> source, FieldHandle<T> target,
  std::unique_ptr<TransferOperator<T>> transfer)
{
  PCMS_ALWAYS_ASSERT(transfer != nullptr);
  const std::size_t id = transfers_.size();
  transfers_.push_back(std::make_unique<BoundTransfer<T>>(
    std::move(source), std::move(target), std::move(transfer)));
  return TransferHandle{this, id};
}

template <typename T>
pcms::FieldHandle<T> pcms::Application::AddField(std::string name,
                                                 Field<T>&& field,
                                                 bool participates)
{
  return AddField(std::move(name), std::move(field),
                  std::make_unique<FieldSerializer<T>>(), participates);
}

template <typename T>
pcms::FieldHandle<T> pcms::Application::AddField(
  std::string name, Field<T>&& field,
  std::unique_ptr<FieldSerializer<T>> serializer, bool participates)
{
  PCMS_FUNCTION_TIMER;
  (void)participates;
  auto [field_it, field_inserted] = fields_.emplace(name, std::move(field));
  if (!field_inserted) {
    throw pcms_error("Field with this name already exists");
  }
  auto& field_obj = std::get<Field<T>>(field_it->second);
  const FieldLayout& layout = field_obj.GetLayout();
  FieldLayoutCommunicator& layout_communicator = GetLayoutCommunicator(layout);
  FieldCommunicator2Ptr field_communicator =
    std::make_unique<FieldCommunicator<T>>(name, layout_communicator, field_obj,
                                           std::move(serializer));

  auto [it, inserted] =
    field_communicators_.emplace(name, std::move(field_communicator));
  if (!inserted) {
    fields_.erase(field_it);
    throw pcms_error("Field with this name already exists");
  }
  return FieldHandle<T>{this, std::move(name)};
}

#endif // COUPLER2_H_
