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
  std::vector<wdmcpl::LO> message_permutation;
  redev::BidirectionalComm<T> comm;
  bool buffer_size_needs_update;
};
// This must be same order as Type enum
using FieldData =
  std::variant<FieldDataT<Real>, FieldDataT<LO>, FieldDataT<GO>>;

struct MeshPartitionData
{
  redev::Redev redev;
  MPI_Comm mpi_comm;
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
  void AddMeshPartition(std::string name, MPI_Comm comm,
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
        .redev = redev::Redev(comm, std::move(partition), process_type_),
        .mpi_comm = comm});
  }

private:
  struct OutMsg
  {
    redev::LOs dest;
    redev::LOs offset;
  };
  // given a partition return a map of ranks and the number of entities on those
  // ranks
  template <typename Func>
  OutMsg construct_out_message(std::string_view name, Field* field,
                               const redev::Partition& partition,
                               Func&& map_func)
  {
    std::map<int, int> dest_rank_counts = map_func(name, field, partition);
    OutMsg out;
    // create dest and offsets arrays from degree array
    out.offset.resize(dest_rank_counts.size() + 1);
    out.dest.resize(dest_rank_counts.size());
    out.offset[0] = 0;
    // isn't this just an exclusive scan for the offset?
    int i = 1;
    for (auto rankCount : dest_rank_counts) {
      out.dest[i - 1] = rankCount.first;
      out.offset[i] = out.offset[i - 1] + rankCount.second;
      i++;
    }
    return out;
  }
  OutMsg construct_out_message(int rank, int nproc,
                               const redev::InMessageLayout& in)
  {

    // auto nAppProcs =
    // Omega_h::divide_no_remainder(in.srcRanks.size(),static_cast<size_t>(nproc));
    auto nAppProcs = in.srcRanks.size() / static_cast<size_t>(nproc);
    // build dest and offsets arrays from incoming message metadata
    redev::LOs senderDeg(nAppProcs);
    for (size_t i = 0; i < nAppProcs - 1; i++) {
      senderDeg[i] =
        in.srcRanks[(i + 1) * nproc + rank] - in.srcRanks[i * nproc + rank];
    }
    const auto totInMsgs = in.offset[rank + 1] - in.offset[rank];
    senderDeg[nAppProcs - 1] =
      totInMsgs - in.srcRanks[(nAppProcs - 1) * nproc + rank];
    OutMsg out;
    for (size_t i = 0; i < nAppProcs; i++) {
      if (senderDeg[i] > 0) {
        out.dest.push_back(i);
      }
    }
    redev::GO sum = 0;
    for (auto deg : senderDeg) { // exscan over values > 0
      if (deg > 0) {
        out.offset.push_back(sum);
        sum += deg;
      }
    }
    out.offset.push_back(sum);
    return out;
  }

public:
  // register_field sets up a rdv::BidirectionalComm on the given mesh rdv
  // object for the field
  template <typename T, typename Func>
  void AddField(std::string_view mesh_partition_name, std::string field_name,
                Field* field, Func&& rank_count_func)
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

    if (process_type_ == redev::ProcessType::Client) {
      auto out_message = construct_out_message(
        field_name, field, mesh_partition.redev.GetPartition(),
        rank_count_func);
      comm.SetOutMessageLayout(out_message.dest, out_message.offset);
    } else {
      int rank, nproc;
      MPI_Comm_rank(mesh_partition.mpi_comm, &rank);
      MPI_Comm_size(mesh_partition.mpi_comm, &nproc);
      auto out_message =
        construct_out_message(rank, nproc, comm.GetInMessageLayout());
      comm.SetOutMessageLayout(out_message.dest, out_message.offset);
    }

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
  void ReceiveField(std::string_view name, Func&& deserializer)
  {
    std::visit(
      [&deserializer, name](auto&& field) {
        auto data = field.comm.Recv();
        const auto buffer =
          nonstd::span<const typename decltype(data)::value_type>(data);
        const auto permutation = nonstd::span<
          const typename decltype(field.message_permutation)::value_type>(
          field.message_permutation);
        // load data into the field based on user specified function/functor
        deserializer(name, field.field, buffer, permutation);
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
  void SendField(std::string_view name, Func&& serializer)
  {
    std::visit(
      [&serializer, name](auto&& field) {
        // TODO: if the field size needs to be updated also need to update the
        // out message layout and permutation arrays
        if (field.buffer_size_needs_update) {
          // pass empty buffer to serializer for first pass of algorithm
          const auto buffer =
            nonstd::span<typename decltype(field.comm_buffer)::value_type>{};
          const auto permutation = nonstd::span<
            const typename decltype(field.message_permutation)::value_type>{};
          auto n = serializer(name, field.field, buffer, permutation);
          field.comm_buffer.resize(n);
        }
        auto buffer = nonstd::span(field.comm_buffer);
        const auto permutation = nonstd::span<
          const typename decltype(field.message_permutation)::value_type>(
          field.message_permutation);
        serializer(name, field.field, buffer, permutation);

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
                   "to AddField)\n";
      std::exit(EXIT_FAILURE);
    }
    return it->second;
  }
  MeshPartitionData& find_mesh_partition_or_error(std::string_view name)
  {
    auto it = mesh_partitions_.find(std::string(name));
    if (it == mesh_partitions_.end()) {
      std::cerr << "Mesh partition must be registered with coupler. (You "
                   "forgot a call to AddMeshPartition)\n";
      std::exit(EXIT_FAILURE);
    }
    return it->second;
  }
};
} // namespace wdmcpl

#endif
