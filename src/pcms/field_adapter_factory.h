#ifndef PCMS_FIELD_ADAPTER_FACTORY_H
#define PCMS_FIELD_ADAPTER_FACTORY_H

#include "pcms/field_adapter_interface.h"
#include "pcms/adapter/xgc/xgc_field_adapter.h"
#include "pcms/dummy_field_adapter.h"
#include "pcms/memory_spaces.h"
#include "pcms/types.h"
#include <memory>
#include <functional>

namespace pcms
{

// Factory functions for creating field adapters with type safety

// XGC Field Adapter factories
template <typename T, typename CoordinateElementType = Real>
std::unique_ptr<XGCFieldAdapter<T, CoordinateElementType>> 
CreateXGCFieldAdapter(std::string name, 
                      MPI_Comm plane_communicator,
                      Rank1View<T, HostMemorySpace> data,
                      const ReverseClassificationVertex& reverse_classification,
                      std::function<int8_t(int, int)> in_overlap) {
    return std::make_unique<XGCFieldAdapter<T, CoordinateElementType>>(
        std::move(name), plane_communicator, data, reverse_classification, in_overlap);
}

// Convenience functions for common types
inline std::unique_ptr<XGCFieldAdapter<float>> 
CreateXGCFloatAdapter(std::string name, 
                      MPI_Comm plane_communicator,
                      Rank1View<float, HostMemorySpace> data,
                      const ReverseClassificationVertex& reverse_classification,
                      std::function<int8_t(int, int)> in_overlap) {
    return CreateXGCFieldAdapter<float>(std::move(name), plane_communicator, 
                                       data, reverse_classification, in_overlap);
}

inline std::unique_ptr<XGCFieldAdapter<double>> 
CreateXGCDoubleAdapter(std::string name, 
                       MPI_Comm plane_communicator,
                       Rank1View<double, HostMemorySpace> data,
                       const ReverseClassificationVertex& reverse_classification,
                       std::function<int8_t(int, int)> in_overlap) {
    return CreateXGCFieldAdapter<double>(std::move(name), plane_communicator, 
                                        data, reverse_classification, in_overlap);
}

// OmegaH Field Adapter factories (when OMEGA_H is enabled)
#ifdef PCMS_HAS_OMEGA_H
#include "pcms/adapter/omega_h/omega_h_field.h"

template <typename T>
std::unique_ptr<OmegaHFieldAdapter<T>> 
CreateOmegaHFieldAdapter(std::string name, 
                        Omega_h::Mesh& mesh,
                        std::string global_id_name = "",
                        int search_nx = 10,
                        int search_ny = 10,
                        mesh_entity_type entity_type = mesh_entity_type::VERTEX) {
    return std::make_unique<OmegaHFieldAdapter<T>>(std::move(name), mesh, 
                                                  std::move(global_id_name),
                                                  search_nx, search_ny, entity_type);
}

inline std::unique_ptr<OmegaHFieldAdapter<float>> 
CreateOmegaHFloatAdapter(std::string name, 
                        Omega_h::Mesh& mesh,
                        std::string global_id_name = "",
                        int search_nx = 10,
                        int search_ny = 10,
                        mesh_entity_type entity_type = mesh_entity_type::VERTEX) {
    return CreateOmegaHFieldAdapter<float>(std::move(name), mesh, std::move(global_id_name),
                                          search_nx, search_ny, entity_type);
}

inline std::unique_ptr<OmegaHFieldAdapter<double>> 
CreateOmegaHDoubleAdapter(std::string name, 
                         Omega_h::Mesh& mesh,
                         std::string global_id_name = "",
                         int search_nx = 10,
                         int search_ny = 10,
                         mesh_entity_type entity_type = mesh_entity_type::VERTEX) {
    return CreateOmegaHFieldAdapter<double>(std::move(name), mesh, std::move(global_id_name),
                                           search_nx, search_ny, entity_type);
}
#endif // PCMS_HAS_OMEGA_H

// Dummy Field Adapter factories
template <typename T>
std::unique_ptr<DummyFieldAdapter<T>> 
CreateDummyFieldAdapter(std::string name = "dummy") {
    return std::make_unique<DummyFieldAdapter<T>>(std::move(name));
}

inline std::unique_ptr<DummyFieldAdapter<float>> 
CreateDummyFloatAdapter(std::string name = "dummy") {
    return CreateDummyFieldAdapter<float>(std::move(name));
}

inline std::unique_ptr<DummyFieldAdapter<double>> 
CreateDummyDoubleAdapter(std::string name = "dummy") {
    return CreateDummyFieldAdapter<double>(std::move(name));
}

// Generic factory that returns base interface pointer
template <typename AdapterType, typename... Args>
std::unique_ptr<IFieldAdapter> 
CreateFieldAdapterGeneric(Args&&... args) {
    static_assert(std::is_base_of_v<IFieldAdapter, AdapterType>, 
                  "AdapterType must inherit from IFieldAdapter");
    return std::make_unique<AdapterType>(std::forward<Args>(args)...);
}

// Factory function that creates adapter based on enum type
template <typename T>
std::unique_ptr<IFieldAdapter> 
CreateFieldAdapterByType(FieldAdapterType type, std::string name) {
    switch (type) {
        case FieldAdapterType::DUMMY_FLOAT:
            if constexpr (std::is_same_v<T, float>) {
                return CreateDummyFloatAdapter(std::move(name));
            } else {
                throw std::runtime_error("Type mismatch: requested DUMMY_FLOAT with non-float type");
            }
        case FieldAdapterType::DUMMY_DOUBLE:
            if constexpr (std::is_same_v<T, double>) {
                return CreateDummyDoubleAdapter(std::move(name));
            } else {
                throw std::runtime_error("Type mismatch: requested DUMMY_DOUBLE with non-double type");
            }
#ifdef PCMS_HAS_OMEGA_H
        case FieldAdapterType::OMEGA_H_FLOAT:
            if constexpr (std::is_same_v<T, float>) {
                throw std::runtime_error("OmegaH adapter creation requires mesh parameter - use factory functions");
            } else {
                throw std::runtime_error("Type mismatch: requested OMEGA_H_FLOAT with non-float type");
            }
        case FieldAdapterType::OMEGA_H_DOUBLE:
            if constexpr (std::is_same_v<T, double>) {
                throw std::runtime_error("OmegaH adapter creation requires mesh parameter - use factory functions");
            } else {
                throw std::runtime_error("Type mismatch: requested OMEGA_H_DOUBLE with non-double type");
            }
#endif
        // Add more cases as other adapters are implemented
        default:
            throw std::runtime_error("Unsupported field adapter type");
    }
}

// Serialization helper functions for working with variants
template <typename T>
SerializationBuffer CreateSerializationBuffer(Rank1View<T, HostMemorySpace> buffer) {
    return MakeSerializationBuffer(buffer);
}

template <typename T>
ConstSerializationBuffer CreateConstSerializationBuffer(Rank1View<const T, HostMemorySpace> buffer) {
    return MakeConstSerializationBuffer(buffer);
}

} // namespace pcms

#endif // PCMS_FIELD_ADAPTER_FACTORY_H