#ifndef PCMS_SRC_PCMS_DUMMY_FIELD_ADAPTER_H
#define PCMS_SRC_PCMS_DUMMY_FIELD_ADAPTER_H
#include "pcms/field_adapter_interface.h"
#include "pcms/memory_spaces.h"
#include "pcms/types.h"
#include "pcms/arrays.h"
#include <string>
#include <vector>

namespace pcms
{

template <typename T = double>
class DummyFieldAdapter : public FieldAdapterBase<DummyFieldAdapter<T>, T>
{
  friend class FieldAdapterBase<DummyFieldAdapter<T>, T>;
public:
  using value_type = T;
  using memory_space = HostMemorySpace;
  
  DummyFieldAdapter(std::string name = "dummy") : name_(std::move(name)) {}

  // Implement required virtual methods from IFieldAdapter
  const std::string& GetName() const override { return name_; }
  
  FieldAdapterType GetAdapterType() const noexcept override {
    if constexpr (std::is_same_v<T, float>) {
      return FieldAdapterType::DUMMY_FLOAT;
    } else {
      return FieldAdapterType::DUMMY_DOUBLE;
    }
  }

  std::unique_ptr<IFieldAdapter> Clone() const override {
    return std::make_unique<DummyFieldAdapter<T>>(*this);
  }

  [[nodiscard]] bool RankParticipatesCouplingCommunication() const noexcept override
  {
    return true; // Dummy adapter always participates
  }

  [[nodiscard]] pcms::mesh_entity_type GetEntityType() const noexcept override
  {
    return pcms::mesh_entity_type::VERTEX;
  }

  [[nodiscard]] std::vector<GO> GetGids() const override
  {
    return {};
  }
  
  [[nodiscard]] ReversePartitionMap GetReversePartitionMap(
    const Partition& partition) const override
  {
    return {};
  }

public:
  // Implement type-safe serialization for CRTP base
  int SerializeImpl(Rank1View<T, memory_space> buffer,
                    Rank1View<const pcms::LO, memory_space> permutation) const override
  {
    return 0; // Dummy implementation
  }
  
  void DeserializeImpl(Rank1View<const T, memory_space> buffer,
                       Rank1View<const pcms::LO, memory_space> permutation) override
  {
    // Dummy implementation - do nothing
  }

private:
  std::string name_;
};

// Type aliases for common instantiations
using DummyFieldAdapterFloat = DummyFieldAdapter<float>;
using DummyFieldAdapterDouble = DummyFieldAdapter<double>;

} // namespace pcms

#endif // PCMS_SRC_PCMS_DUMMY_FIELD_ADAPTER_H