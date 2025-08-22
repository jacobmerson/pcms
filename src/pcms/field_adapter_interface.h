#ifndef PCMS_FIELD_ADAPTER_INTERFACE_H
#define PCMS_FIELD_ADAPTER_INTERFACE_H

#include "pcms/types.h"
#include "pcms/arrays.h"
#include "pcms/memory_spaces.h"
#include "pcms/partition.h"
#include "pcms/common.h"
#include <memory>
#include <string>
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <variant>

namespace pcms
{

// Forward declare mesh entity type to avoid circular dependencies
enum class mesh_entity_type : int{
  VERTEX = 0,
  EDGE = 1,
  FACE = 2,
  REGION = 3
};

inline int mesh_entity_to_int(mesh_entity_type entity_type)
{
  static_assert(std::is_same<std::underlying_type_t<mesh_entity_type>, int>::value, "mesh_entity_type must be an int");
  return static_cast<std::underlying_type_t<mesh_entity_type>>(entity_type);
}

// Enum for adapter types
enum class FieldAdapterType {
    XGC_FLOAT,
    XGC_DOUBLE,
    XGC_INT,
    XGC_LONG_INT,
    OMEGA_H_FLOAT,
    OMEGA_H_DOUBLE,
    DUMMY_FLOAT,
    DUMMY_DOUBLE,
    POINT_CLOUD_FLOAT,
    POINT_CLOUD_DOUBLE
};

// Variant-based buffer for serialization
using SerializationBuffer = std::variant<
    Rank1View<float, HostMemorySpace>,
    Rank1View<double, HostMemorySpace>,
    Rank1View<int, HostMemorySpace>,
    Rank1View<int64_t, HostMemorySpace>
>;

using ConstSerializationBuffer = std::variant<
    Rank1View<const float, HostMemorySpace>,
    Rank1View<const double, HostMemorySpace>, 
    Rank1View<const int, HostMemorySpace>,
    Rank1View<const int64_t, HostMemorySpace>
>;

// Base interface that all field adapters must implement
class IFieldAdapter {
public:
    virtual ~IFieldAdapter() = default;
    
    // Core field operations
    virtual const std::string& GetName() const = 0;
    virtual mesh_entity_type GetEntityType() const noexcept = 0;
    virtual bool RankParticipatesCouplingCommunication() const noexcept = 0;
    
    // Variant-based serialization interface
    virtual int Serialize(SerializationBuffer buffer,
                         Rank1View<const LO, HostMemorySpace> permutation) const = 0;
    virtual void Deserialize(ConstSerializationBuffer buffer,
                            Rank1View<const LO, HostMemorySpace> permutation) = 0;
    
    // Partitioning
    virtual std::vector<GO> GetGids() const = 0;
    virtual ReversePartitionMap GetReversePartitionMap(const Partition& partition) const = 0;
    
    // Type identification
    virtual FieldAdapterType GetAdapterType() const noexcept = 0;
    virtual const std::type_info& GetValueType() const noexcept = 0;
    
    // Clone support for copying
    virtual std::unique_ptr<IFieldAdapter> Clone() const = 0;
};

// CRTP base to reduce boilerplate and provide type safety
template <typename Derived, typename ValueType>
class FieldAdapterBase : public IFieldAdapter {
public:
    using value_type = ValueType;
    using memory_space = HostMemorySpace;
    
    // Implement variant-based serialization dispatch
    int Serialize(SerializationBuffer buffer,
                 Rank1View<const LO, HostMemorySpace> permutation) const final {
        return std::visit([this, &permutation](auto& typed_buffer) -> int {
            using BufferValueType = typename std::decay_t<decltype(typed_buffer)>::value_type;
            if constexpr (std::is_same_v<BufferValueType, ValueType>) {
                return static_cast<const Derived*>(this)->SerializeImpl(typed_buffer, permutation);
            } else {
                throw std::runtime_error("Serialize buffer type mismatch: expected " + 
                                       std::string(typeid(ValueType).name()) + 
                                       ", got " + std::string(typeid(BufferValueType).name()));
            }
        }, buffer);
    }
    
    void Deserialize(ConstSerializationBuffer buffer,
                    Rank1View<const LO, HostMemorySpace> permutation) final {
        std::visit([this, &permutation](auto& typed_buffer) {
            using BufferValueType = typename std::decay_t<decltype(typed_buffer)>::value_type;
            if constexpr (std::is_same_v<BufferValueType, ValueType>) {
                static_cast<Derived*>(this)->DeserializeImpl(typed_buffer, permutation);
            } else {
                throw std::runtime_error("Deserialize buffer type mismatch: expected " + 
                                       std::string(typeid(ValueType).name()) + 
                                       ", got " + std::string(typeid(BufferValueType).name()));
            }
        }, buffer);
    }
    
    const std::type_info& GetValueType() const noexcept final {
        return typeid(ValueType);
    }

protected:
    // Derived classes implement these type-safe methods
    virtual int SerializeImpl(Rank1View<ValueType, HostMemorySpace> buffer,
                             Rank1View<const LO, HostMemorySpace> permutation) const = 0;
    virtual void DeserializeImpl(Rank1View<const ValueType, HostMemorySpace> buffer,
                                Rank1View<const LO, HostMemorySpace> permutation) = 0;
};

// Helper functions to create buffers of the right type
template <typename T>
SerializationBuffer MakeSerializationBuffer(Rank1View<T, HostMemorySpace> buffer) {
    return SerializationBuffer{buffer};
}

template <typename T>  
ConstSerializationBuffer MakeConstSerializationBuffer(Rank1View<const T, HostMemorySpace> buffer) {
    return ConstSerializationBuffer{buffer};
}

} // namespace pcms

#endif // PCMS_FIELD_ADAPTER_INTERFACE_H