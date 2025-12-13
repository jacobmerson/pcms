#ifndef PCMS_FIELD_LAYOUT_H
#define PCMS_FIELD_LAYOUT_H
#include <array>
#include <map>
#include <memory>
#include <vector>
#include <redev.h>
#include "pcms/field.h"
#include "pcms/arrays.h"
#include "pcms/coordinate_system.h"

namespace pcms
{

constexpr int ent_offsets_len = 5;
using EntOffsetsArray = std::array<size_t, ent_offsets_len>;

struct FieldLayoutPlan
{
  redev::LOs destinations;
  redev::LOs offsets;
  std::vector<pcms::LO> permutation;
  std::vector<pcms::GO> gid_payload;
};

using ReversePartitionMap = std::map<pcms::LO, std::vector<pcms::LO>>;

template <typename T>
class FieldT;

class FieldLayout
{
public:
  virtual std::unique_ptr<FieldT<Real>> CreateField() const = 0;

  // number of components
  int virtual GetNumComponents() const = 0;

  // nodes for standard lagrange FEM
  LO virtual GetNumOwnedDofHolder() const = 0;
  GO virtual GetNumGlobalDofHolder() const = 0;

  // size of buffer that needs to be allocated to represent the field
  // # components * NumDOFHolder
  LO OwnedSize() const { return GetNumComponents() * GetNumOwnedDofHolder(); };
  GO GlobalSize() const
  {
    return GetNumComponents() * GetNumGlobalDofHolder();
  };

  virtual Rank1View<const bool, HostMemorySpace> GetOwned() const = 0;
  virtual GlobalIDView<HostMemorySpace> GetGids() const = 0;

  // returns true if the field layout is distributed
  // if the field layout is distributed, the owned and global dofs are the same
  virtual bool IsDistributed() = 0;

  // This class should construct the permutation arrays that are needed
  // for serialization / deserialization
  //

  virtual EntOffsetsArray GetEntOffsets() const = 0;

  virtual FieldLayoutPlan BuildClientPlan(const redev::Partition& partition) const = 0;

  virtual FieldLayoutPlan BuildServerPlan(
    GlobalIDView<HostMemorySpace> received_gids,
    const redev::InMessageLayout& incoming_layout, int mpi_rank,
    int mpi_size) const = 0;

  virtual CoordinateView<HostMemorySpace> GetDOFHolderCoordinates() const = 0;

  // Serialize, Derserialize, ReversePartitionMap?
  // GetOwnedDofHolderCoordinates(CoordinateSystem);

  // Serialize(FieldDataView, SerializationBuffer);
  // Deserialize(SerializationBuffer, FieldDataView);

  // Adjacency information
  // TODO: Need a GraphView class (simply two rank1 arrays CSR matrix w/o
  // values) virtual bool HasAdjacency() = 0; virtual GraphView GetAdjacency(LO
  // dim) = 0;

  virtual ~FieldLayout() noexcept = default;
};

} // namespace pcms
#endif // PCMS_FIELD_LAYOUT_H
