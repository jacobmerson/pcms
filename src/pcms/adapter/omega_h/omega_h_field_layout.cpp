#include "omega_h_field.h"
#include "omega_h_field2.h"
#include "pcms/adapter/omega_h/omega_h_field_layout.h"
#include "omega_h_field_layout.h"
#include "pcms/inclusive_scan.h"
#include "pcms/profile.h"
#include <iterator>
#include <map>
#include <memory>

namespace pcms
{
/*
 * Field Layout
 */

namespace
{
struct PartitionMapping
{
  std::vector<pcms::LO> indices;
  EntOffsetsArray ent_offsets;
  PartitionMapping() { ent_offsets.fill(0); }
};

using DestinationMap = std::map<pcms::LO, PartitionMapping>;

struct OutgoingMessage
{
  redev::LOs destinations;
  redev::LOs offsets;
};

OutgoingMessage BuildOutgoingMessage(const DestinationMap& reverse_partition)
{
  PCMS_FUNCTION_TIMER;
  OutgoingMessage out;
  redev::LOs counts;
  counts.reserve(reverse_partition.size());
  out.destinations.clear();
  out.destinations.reserve(reverse_partition.size());
  for (const auto& rank : reverse_partition) {
    out.destinations.push_back(rank.first);
    const auto num_indices = rank.second.indices.size();
    counts.push_back(num_indices + rank.second.ent_offsets.size());
  }
  out.offsets.resize(counts.size() + 1);
  out.offsets[0] = 0;
  pcms::inclusive_scan(counts.begin(), counts.end(),
                       std::next(out.offsets.begin(), 1));
  return out;
}

std::vector<pcms::LO> BuildClientPermutation(const DestinationMap& reverse_partition,
                                             size_t num_entries, int* length)
{
  PCMS_FUNCTION_TIMER;
  std::vector<pcms::LO> permutation(num_entries);
  pcms::LO entry = 0;
  for (const auto& rank : reverse_partition) {
    entry += ent_offsets_len;
    for (int e = 0; e < static_cast<int>(rank.second.ent_offsets.size()) - 1;
         ++e) {
      const auto start = rank.second.ent_offsets[e];
      const auto end = rank.second.ent_offsets[e + 1];
      for (size_t i = start; i < end; ++i) {
        const auto index = rank.second.indices[i];
        PCMS_ALWAYS_ASSERT(static_cast<size_t>(index) < permutation.size());
        permutation[static_cast<size_t>(index)] = entry++;
      }
    }
  }
  *length = entry;
  return permutation;
}

OutgoingMessage BuildServerOutgoingMessage(
  int rank, int nproc, const redev::InMessageLayout& in)
{
  PCMS_FUNCTION_TIMER;
  OutgoingMessage out;
  REDEV_ALWAYS_ASSERT(!in.srcRanks.empty());
  auto nAppProcs = in.srcRanks.size() / static_cast<size_t>(nproc);
  redev::LOs senderDeg(nAppProcs);
  for (size_t i = 0; i < nAppProcs - 1; i++) {
    senderDeg[i] = in.srcRanks[(i + 1) * nproc + rank] - in.srcRanks[i * nproc + rank];
  }
  const auto totInMsgs = in.offset[rank + 1] - in.offset[rank];
  senderDeg[nAppProcs - 1] =
    totInMsgs - in.srcRanks[(nAppProcs - 1) * nproc + rank];
  for (size_t i = 0; i < nAppProcs; i++) {
    if (senderDeg[i] > 0) {
      out.destinations.push_back(i);
    }
  }
  redev::GO sum = 0;
  for (auto deg : senderDeg) {
    if (deg > 0) {
      out.offsets.push_back(sum);
      sum += deg;
    }
  }
  out.offsets.push_back(sum);
  return out;
}

std::vector<pcms::LO> BuildServerPermutation(
  GlobalIDView<HostMemorySpace> local_gids,
  GlobalIDView<HostMemorySpace> received_msg, EntOffsetsArray ent_offsets)
{
  PCMS_FUNCTION_TIMER;
  std::array<std::map<pcms::GO, pcms::LO>, ent_offsets_len - 1> gid_to_buffer_index;
  size_t offset = 0;
  while (offset < received_msg.size()) {
    GlobalIDView<HostMemorySpace> received_offsets(
      received_msg.data_handle() + offset, ent_offsets_len);
    const auto length = received_offsets[received_offsets.size() - 1];
    GlobalIDView<HostMemorySpace> received_gids(
      received_msg.data_handle() + offset + ent_offsets_len, length);
    PCMS_ALWAYS_ASSERT(offset + ent_offsets_len + length - 1 < received_msg.size());
    for (int e = 0; e < received_offsets.size() - 1; ++e) {
      auto start = received_offsets[e];
      auto end = received_offsets[e + 1];
      for (auto i = start; i < end; ++i) {
        gid_to_buffer_index[e][received_gids[i]] =
          offset + ent_offsets_len + i;
      }
    }
    offset += length + ent_offsets_len;
  }
  std::vector<pcms::LO> permutation;
  permutation.reserve(local_gids.size());
  for (int e = 0; e < ent_offsets.size() - 1; ++e) {
    auto start = ent_offsets[e];
    auto end = ent_offsets[e + 1];
    for (auto i = start; i < end; ++i) {
      permutation.push_back(gid_to_buffer_index[e][local_gids[i]]);
    }
  }
  REDEV_ALWAYS_ASSERT(permutation.size() == local_gids.size());
  return permutation;
}

DestinationMap BuildDestinationMap(const OmegaHFieldLayout& layout,
                                   const redev::Partition& partition)
{
  PCMS_FUNCTION_TIMER;
  DestinationMap reverse_partition;
  auto classIds_h = Omega_h::HostRead<Omega_h::ClassId>(layout.GetClassIDs());
  auto classDims_h = Omega_h::HostRead<Omega_h::I8>(layout.GetClassDims());
  auto owned = layout.GetOwned();
  const auto coords = layout.GetDOFHolderCoordinates().GetCoordinates();
  auto dim = layout.GetMesh().dim();
  const auto nodes_per_dim = layout.GetNodesPerDim();
  pcms::LO local_index = 0;
  for (int ent_dim = 0; ent_dim <= layout.GetMesh().dim(); ++ent_dim) {
    if (nodes_per_dim[ent_dim] == 0)
      continue;
    for (pcms::LO i = 0; i < layout.GetMesh().nents(ent_dim); ++i, ++local_index) {
      if (!owned[local_index])
        continue;
      std::array<pcms::Real, 3> coord{};
      coord[0] = coords(local_index, 0);
      coord[1] = coords(local_index, 1);
      coord[2] = (dim > 2) ? coords(local_index, 2) : 0.0;
      auto dest_rank =
        std::visit(GetRank{classIds_h[local_index], classDims_h[local_index], coord},
                   partition);
      auto& mapping = reverse_partition[dest_rank];
      mapping.indices.emplace_back(local_index);
      const auto n = mapping.ent_offsets.size();
      for (size_t e = ent_dim + 1; e < n; ++e) {
        mapping.ent_offsets[e] += 1;
      }
    }
  }
  return reverse_partition;
}
} // namespace

template <typename T>
Omega_h::Write<Omega_h::GO> GetGidsHelper(LO total_ents,
                                          std::array<int, 4> nodes_per_dim,
                                          Omega_h::Mesh& mesh,
                                          const std::string& global_id_name)
{
  PCMS_FUNCTION_TIMER;

  Omega_h::Write<Omega_h::GO> owned_gids(total_ents);
  LO offset = 0;
  for (int i = 0; i <= mesh.dim(); ++i) {
    if (nodes_per_dim[i]) {
      auto dim_gids = mesh.get_array<T>(i, global_id_name);
      Omega_h::parallel_for(
        dim_gids.size(),
        OMEGA_H_LAMBDA(int i) { owned_gids[i + offset] = dim_gids[i]; });
      offset += dim_gids.size();
    }
  }

  PCMS_ALWAYS_ASSERT(offset == total_ents);

  return owned_gids;
}

// this is a workaround to specify the parametric coordinates for MeshFields to
// be replaced when https://github.com/SCOREC/meshFields/issues/70 is resolved
struct ComputeVertexCoordsFunctor
{
  Kokkos::View<Real**> dof_holder_coords_;
  Omega_h::Reals coords_;
  size_t offset_;

  ComputeVertexCoordsFunctor(Kokkos::View<Real**> dof_holder_coords,
                             Omega_h::Reals coords, size_t offset)
    : dof_holder_coords_(dof_holder_coords), coords_(coords), offset_(offset)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(LO i) const
  {
    dof_holder_coords_(offset_ + i, 0) = coords_[2 * i + 0];
    dof_holder_coords_(offset_ + i, 1) = coords_[2 * i + 1];
  }
};

// this is a workaround to specify the parametric coordinates for MeshFields to
// be replaced when https://github.com/SCOREC/meshFields/issues/70 is resolved
struct ComputeEdgeCoordsFunctor
{
  Kokkos::View<Real**> dof_holder_coords_;
  Omega_h::Reals coords_;
  Omega_h::LOs edge_verts_;
  size_t offset_;

  ComputeEdgeCoordsFunctor(Kokkos::View<Real**> dof_holder_coords,
                           Omega_h::Reals coords, Omega_h::LOs edge_verts,
                           size_t offset)
    : dof_holder_coords_(dof_holder_coords),
      coords_(coords),
      edge_verts_(edge_verts),
      offset_(offset)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(LO i) const
  {
    auto verts = Omega_h::gather_verts<2>(edge_verts_, i);
    Real x0 = coords_[2 * verts[0] + 0];
    Real y0 = coords_[2 * verts[0] + 1];
    Real x1 = coords_[2 * verts[1] + 0];
    Real y1 = coords_[2 * verts[1] + 1];
    dof_holder_coords_(offset_ + i, 0) = (x0 + x1) / 2;
    dof_holder_coords_(offset_ + i, 1) = (y0 + y1) / 2;
  }
};

struct CopyClassInfoFunctor
{
  Omega_h::Write<Omega_h::ClassId> class_ids_;
  Omega_h::Write<Omega_h::I8> class_dims_;
  Kokkos::View<bool*> owned_;
  Omega_h::Read<Omega_h::ClassId> ids_;
  Omega_h::Read<Omega_h::I8> dims_;
  Omega_h::Read<Omega_h::I8> owned_data_;
  size_t offset_;

  CopyClassInfoFunctor(Omega_h::Write<Omega_h::ClassId> class_ids,
                       Omega_h::Write<Omega_h::I8> class_dims,
                       Kokkos::View<bool*> owned,
                       Omega_h::Read<Omega_h::ClassId> ids,
                       Omega_h::Read<Omega_h::I8> dims,
                       Omega_h::Read<Omega_h::I8> owned_data, size_t offset)
    : class_ids_(class_ids),
      class_dims_(class_dims),
      owned_(owned),
      ids_(ids),
      dims_(dims),
      owned_data_(owned_data),
      offset_(offset)
  {
  }

  OMEGA_H_DEVICE
  void operator()(LO i) const
  {
    class_ids_[offset_ + i] = ids_[i];
    class_dims_[offset_ + i] = dims_[i];
    owned_[offset_ + i] = owned_data_[i];
  }
};

OmegaHFieldLayout::OmegaHFieldLayout(Omega_h::Mesh& mesh,
                                     std::array<int, 4> nodes_per_dim,
                                     int num_components,
                                     CoordinateSystem coordinate_system,
                                     std::string global_id_name)
  : mesh_(mesh),
    global_id_name_(global_id_name),
    num_components_(num_components),
    coordinate_system_(coordinate_system),
    nodes_per_dim_(nodes_per_dim),
    dof_holder_coords_("", GetNumOwnedDofHolder(), mesh_.dim()),
    dof_holder_coords_host_("dof_holder_coords_host", GetNumOwnedDofHolder(),
                            mesh_.dim()),
    class_ids_(GetNumEnts()),
    class_dims_(class_ids_.size()),
    owned_("", class_dims_.size()),
    owned_host_("", class_dims_.size())
{
  PCMS_FUNCTION_TIMER;
  LO total_ents = GetNumEnts();

  auto tag = mesh_.get_tagbase(0, global_id_name_);
  if (Omega_h::is<GO>(tag)) {
    gids_ = GetGidsHelper<GO>(total_ents, nodes_per_dim, mesh, global_id_name);
  } else if (Omega_h::is<LO>(tag)) {
    gids_ = GetGidsHelper<LO>(total_ents, nodes_per_dim, mesh, global_id_name);
  } else {
    std::cerr << "Weird tag type for global arrays.\n";
    std::abort();
  }

  auto coords = mesh_.coords();

  size_t offset = 0;
  for (int i = 0; i <= mesh_.dim(); ++i) {
    if (nodes_per_dim[i] == 1) {
      if (i == 0) {
        ComputeVertexCoordsFunctor functor(dof_holder_coords_, coords, offset);
        Kokkos::parallel_for(mesh_.nents(0), functor);
      } else if (i == 1) {
        auto edge_verts = mesh_.ask_verts_of(1);
        ComputeEdgeCoordsFunctor functor(dof_holder_coords_, coords, edge_verts,
                                         offset);
        Kokkos::parallel_for(mesh_.nents(1), functor);
      } else {
        std::cerr << "Unsupported" << std::endl;
        std::abort();
      }
    } else if (nodes_per_dim[i] != 0) {
      std::cerr << "Unsupported" << std::endl;
      std::abort();
    }

    offset += mesh.nents(i);
  }

  offset = 0;
  for (int i = 0; i <= mesh_.dim(); ++i) {
    if (nodes_per_dim_[i]) {
      auto ids = mesh_.get_array<Omega_h::ClassId>(i, "class_id");
      auto dims = mesh_.get_array<Omega_h::I8>(i, "class_dim");
      auto owned = mesh_.owned(i);
      PCMS_ALWAYS_ASSERT(ids.size() == dims.size() &&
                         dims.size() == mesh_.nents(i));

      CopyClassInfoFunctor functor(class_ids_, class_dims_, owned_, ids, dims,
                                   owned, offset);
      Omega_h::parallel_for(mesh_.nents(i), functor);
      offset += mesh.nents(i);
    }
  }
  gids_host_ = Omega_h::HostWrite<Omega_h::GO>(gids_);
}

std::unique_ptr<FieldT<Real>> OmegaHFieldLayout::CreateField() const
{
  return std::make_unique<OmegaHField2>(*this);
}

int OmegaHFieldLayout::GetNumComponents() const
{
  return num_components_;
}

LO OmegaHFieldLayout::GetNumOwnedDofHolder() const
{
  LO count = 0;
  for (int i = 0; i <= mesh_.dim(); ++i) {
    count += mesh_.nents(i) * nodes_per_dim_[i];
  }
  return count;
}

GO OmegaHFieldLayout::GetNumGlobalDofHolder() const
{
  LO count = 0;
  for (int i = 0; i <= mesh_.dim(); ++i) {
    count += mesh_.nglobal_ents(i) * nodes_per_dim_[i];
  }
  return count;
}

std::array<int, 4> OmegaHFieldLayout::GetNodesPerDim() const
{
  return nodes_per_dim_;
}

Rank1View<const bool, HostMemorySpace> OmegaHFieldLayout::GetOwned() const
{
  Kokkos::deep_copy(owned_host_, owned_);
  return make_const_array_view(owned_host_);
}

GlobalIDView<HostMemorySpace> OmegaHFieldLayout::GetGids() const
{
  return GlobalIDView<HostMemorySpace>(gids_host_.data(), gids_host_.size());
}

CoordinateView<HostMemorySpace> OmegaHFieldLayout::GetDOFHolderCoordinates()
  const
{
  deep_copy_mismatch_layouts(dof_holder_coords_host_, dof_holder_coords_);
  Rank2View<const Real, HostMemorySpace> coords_view(
    dof_holder_coords_host_.data(), dof_holder_coords_host_.extent(0), 2);
  return CoordinateView<HostMemorySpace>{coordinate_system_, coords_view};
}

bool OmegaHFieldLayout::IsDistributed()
{
  return true;
}

Omega_h::Read<Omega_h::ClassId> OmegaHFieldLayout::GetClassIDs() const
{
  PCMS_FUNCTION_TIMER;
  return Omega_h::Read(class_ids_);
}

Omega_h::Read<Omega_h::I8> OmegaHFieldLayout::GetClassDims() const
{
  PCMS_FUNCTION_TIMER;
  return Omega_h::Read(class_dims_);
}

size_t OmegaHFieldLayout::GetNumEnts() const
{
  size_t n = 0;
  for (int i = 0; i <= mesh_.dim(); ++i) {
    if (nodes_per_dim_[i])
      n += mesh_.nents(i);
  }
  return n;
}

Omega_h::Mesh& OmegaHFieldLayout::GetMesh() const
{
  return mesh_;
}

EntOffsetsArray OmegaHFieldLayout::GetEntOffsets() const
{
  EntOffsetsArray offsets{};
  size_t offset = 0;
  for (size_t i = 0; i < offsets.size(); ++i) {
    offsets[i] = offset;
    if (i <= static_cast<size_t>(mesh_.dim()) && nodes_per_dim_[i])
      offset += mesh_.nents(i);
  }
  offsets[offsets.size() - 1] = offset;
  return offsets;
}

FieldLayoutPlan OmegaHFieldLayout::BuildClientPlan(
  const redev::Partition& partition) const
{
  PCMS_FUNCTION_TIMER;
  FieldLayoutPlan plan;
  const auto reverse_partition = BuildDestinationMap(*this, partition);
  auto out_message = BuildOutgoingMessage(reverse_partition);
  plan.destinations = std::move(out_message.destinations);
  plan.offsets = std::move(out_message.offsets);

  auto gids = GetGids();
  auto owned = GetOwned();
  int message_length = 0;
  plan.permutation =
    BuildClientPermutation(reverse_partition, gids.size(), &message_length);
  plan.gid_payload.resize(static_cast<size_t>(message_length));
  for (size_t i = 0; i < gids.size(); ++i) {
    if (owned[i]) {
      const auto message_index =
        static_cast<size_t>(plan.permutation[static_cast<size_t>(i)]);
      plan.gid_payload[message_index] = gids[i];
    }
  }

  const auto ent_offsets = GetEntOffsets();
  for (const auto& rank : reverse_partition) {
    if (rank.second.indices.empty())
      continue;
    const auto first_index = static_cast<size_t>(rank.second.indices.front());
    const auto message_index =
      static_cast<size_t>(plan.permutation[first_index]);
    PCMS_ALWAYS_ASSERT(message_index >= ent_offsets_len);
    auto offset_index = message_index - ent_offsets_len;
    for (int i = 0; i < ent_offsets_len; ++i) {
      plan.gid_payload[offset_index + static_cast<size_t>(i)] =
        rank.second.ent_offsets[i];
    }
  }
  return plan;
}

FieldLayoutPlan OmegaHFieldLayout::BuildServerPlan(
  GlobalIDView<HostMemorySpace> received_gids,
  const redev::InMessageLayout& incoming_layout, int mpi_rank,
  int mpi_size) const
{
  PCMS_FUNCTION_TIMER;
  FieldLayoutPlan plan;
  auto out_message = BuildServerOutgoingMessage(mpi_rank, mpi_size, incoming_layout);
  plan.destinations = std::move(out_message.destinations);
  plan.offsets = std::move(out_message.offsets);
  auto gids = GetGids();
  plan.permutation =
    BuildServerPermutation(gids, received_gids, GetEntOffsets());
  plan.gid_payload.clear();
  return plan;
}

} // namespace pcms
