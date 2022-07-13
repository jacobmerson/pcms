#ifndef WDMCPL_H_
#define WDMCPL_H_
#include <mpi.h>
#include <redev.h>
#include "wdmcpl/coordinate_systems.h"
#include <unordered_map>
#include "wdmcpl/assert.h"
#include "wdmcpl/external/span.h"
#include "wdmcpl/types.h"
#include <variant>

namespace wdmcpl
{

// Span cast is needed to go to/from buffers to typed data
template <typename T, typename T2>
std::span<T> span_cast(std::span<T2> spn)
{
  return {reinterpret_cast<T*>(spn.data()), spn.size_bytes() / sizeof(T)};
}
template <typename T, typename T2>
std::span<const T> span_cast(std::span<const T2> spn)
{
  return {reinterpret_cast<const T*>(spn.data()), spn.size_bytes() / sizeof(T)};
}

using ProcessType = redev::ProcessType;
enum class MeshType
{
  Structured,
  Unstructured
};
class Field
{};

class Mesh
{
  MeshType type_;
  std::unordered_map<std::string, Field> fields;
};

template <typename T>
struct FieldDataT
{
  Field* field;
  std::vector<T> comm_buffer;
  redev::BidirectionalComm<T> comm;
  bool buffer_size_needs_update;
};
// This must be same order as Type enum
using FieldData =
  std::variant<FieldDataT<Real>, FieldDataT<LO>, FieldDataT<GO>>;

struct MeshPartitionData
{
  redev::Redev redev;
};
// Coupler needs to have both a standalone mesh definitions to setup rdv comms
// and a list of fields
// in the server it also needs sets of fields that will be combined
class Coupler
{
public:
  Coupler(std::string name, ProcessType process_type)
    : name_(std::move(name)), process_type_(process_type)
  {
  }
  // register mesh sets up the rdv object for the mesh in question including
  // the partitioning
  void add_mesh_partition(std::string name, MPI_Comm comm,
                          redev::Partition partition)
  {
    auto it = mesh_partitions_.find(name);
    if (it != mesh_partitions_.end()) {
      std::cerr << "Mesh with this name" << name << "already exists!\n";
      std::exit(EXIT_FAILURE);
    }
    mesh_partitions_.emplace(
      std::move(name),
      MeshPartitionData{
        .redev = redev::Redev(comm, std::move(partition), process_type_)});
  }

  // register_field sets up a rdv::BidirectionalComm on the given mesh rdv
  // object for the field
  template <typename T>
  void add_field(std::string_view mesh_partition_name, std::string field_name,
                 Field* field)
  {
    auto it = fields_.find(field_name);
    if (it != fields_.end()) {
      std::cerr << "Field with this name" << field_name << "already exists!\n";
      std::exit(EXIT_FAILURE);
    }
    auto& mesh_partition = find_mesh_partition_or_error(mesh_partition_name);
    adios2::Params params;
    auto transport_type = redev::TransportType::BP4;
    std::string transport_name = name_;
    transport_name.append(mesh_partition_name).append(field_name);
    auto comm = mesh_partition.redev.CreateAdiosClient<T>(
      transport_name, params, transport_type);
    fields_.emplace(std::move(field_name), FieldDataT<T>{
                                             .field = field,
                                             .comm_buffer = {},
                                             .comm = std::move(comm),
                                             .buffer_size_needs_update = true,
                                           });
  }
  /**
   * Gather the field from the rendezvous server and deserialize it into the
   * local field definition
   * @tparam Func invocable that takes Field* and span<byte> (buffer to fill
   * field field data)
   * @param name
   * @param deserializer
   */
  template <typename Func>
  void GatherField(std::string_view name, Func&& deserializer)
  {
    std::visit(
      [&deserializer,name](auto&& field) {
        auto data = field.comm.Recv();
        const auto buffer =
          nonstd::span<const typename decltype(data)::value_type>(data);
        // load data into the field based on user specified function/functor
        deserializer(name, field.field, buffer);
      },
      find_field_or_error(name));
  }
  /**
   * Scatter the field to the rendezvous server by serializing the local field
   * definition
   * @tparam Func
   * @param name
   * @param serializer the serializer is an invocable that takes a Field* and a
   * span of one of the wdmcpl data types. It is used in a two pass algorithm,
   * so the implimenter must check the size of the buffer that is passed in. If
   * the buffer has a size of 0, routine must not write any data into the
   * buffer. The serializer invocable must return the number of entries to be
   * serialized. Note, this algorithm is not guaranteed to call the first
   * counting pass.
   */
  template <typename Func>
  void ScatterField(std::string_view name, Func&& serializer)
  {
    std::visit(
      [&serializer,name](auto&& field) {
        if (field.buffer_size_needs_update) {
          // pass empty buffer to serializer for first pass of algorithm
          const auto buffer =
            nonstd::span<typename decltype(field.comm_buffer)::value_type>{};
          auto n = serializer(name, field.field, buffer);
          field.comm_buffer.resize(n);
        }
        auto buffer = nonstd::span(field.comm_buffer);
        serializer(name, field.field, buffer);

        field.comm.Send(field.comm_buffer.data());
      },
      find_field_or_error(name));
  }

private:
  std::string name_;
  ProcessType process_type_;
  std::unordered_map<std::string, FieldData> fields_;
  std::unordered_map<std::string, MeshPartitionData> mesh_partitions_;
  FieldData& find_field_or_error(std::string_view name)
  {
    auto it = fields_.find(std::string(name));
    if (it == fields_.end()) {
      std::cerr << "Field must be registered with coupler. (You forgot a call "
                   "to add_field)\n";
      std::exit(EXIT_FAILURE);
    }
    return it->second;
  }
  MeshPartitionData& find_mesh_partition_or_error(std::string_view name)
  {
    auto it = mesh_partitions_.find(std::string(name));
    if (it == mesh_partitions_.end()) {
      std::cerr << "Mesh partition must be registered with coupler. (You "
                   "forgot a call to add_mesh_partition)\n";
      std::exit(EXIT_FAILURE);
    }
    return it->second;
  }
};
} // namespace wdmcpl

#endif
