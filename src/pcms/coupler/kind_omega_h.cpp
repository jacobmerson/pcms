// The Lagrange spaces of order 1 on Omega_h simplex meshes as a described
// kind. Each application rank sends its owned cells, with the vertices of
// their closure, to the server ranks the partition assigns the cells to. A
// vertex is identified by its owner: the application rank that owns it and its
// index among that rank's owned vertices, from the mesh's remotes or, when the
// application only has global ids, from those. The server numbers the
// vertices from their owners and rebuilds a distributed mesh.

#include "description.hpp"
#include "pcms/backends/omega_h/omega_h.hpp"
#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/field/function_spaces.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/mesh_geometry.hpp"
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <algorithm>
#include <map>

namespace pcms::detail
{

namespace
{

using omega_h::OmegaHGeometry;

/// Header of a block: the sending rank's owned vertex and cell counts and the
/// block's cell and vertex counts.
constexpr size_t HeaderSize = 4;

/// The owner of each local vertex: the application rank that owns it and its
/// index among that rank's owned vertices.
struct VertexOwners
{
  std::vector<GO> ranks;
  std::vector<GO> indices;
};

/// Each local vertex's index among this rank's owned vertices, -1 for the
/// vertices it does not own.
std::vector<LO> OwnedIndices(const std::vector<bool>& owned)
{
  std::vector<LO> indices(owned.size(), -1);
  LO next = 0;
  for (size_t v = 0; v < owned.size(); ++v) {
    if (owned[v]) {
      indices[v] = next++;
    }
  }
  return indices;
}

/// The owners from the mesh's remotes: the mesh's ranks are the application
/// ranks.
VertexOwners OwnersFromRemotes(Omega_h::Mesh& mesh,
                               const std::vector<LO>& owned_indices)
{
  const auto owners = mesh.ask_owners(0);
  Omega_h::HostWrite<LO> indices(static_cast<LO>(owned_indices.size()));
  for (size_t v = 0; v < owned_indices.size(); ++v) {
    indices[static_cast<LO>(v)] = owned_indices[v];
  }
  // copies take the owner's index
  const Omega_h::HostRead<LO> synced(
    mesh.sync_array(0, Omega_h::Read<LO>(indices.write()), 1));
  const Omega_h::HostRead<Omega_h::I32> ranks(owners.ranks);
  VertexOwners result;
  result.ranks.resize(owned_indices.size());
  result.indices.resize(owned_indices.size());
  for (size_t v = 0; v < owned_indices.size(); ++v) {
    result.ranks[v] = ranks[static_cast<LO>(v)];
    result.indices[v] = synced[static_cast<LO>(v)];
  }
  return result;
}

/// The owners from global ids: each id is looked up at the application rank
/// id % size, where its owner registered it.
VertexOwners OwnersFromGlobalIds(const std::vector<GO>& gids,
                                 const std::vector<LO>& owned_indices,
                                 const SetupContext& context)
{
  const int size = context.size;
  // entries (gid, owner rank or -1, owned index or -1) by directory rank
  std::vector<int> send_counts(size, 0);
  for (const GO gid : gids) {
    send_counts[static_cast<size_t>(gid % size)] += 3;
  }
  std::vector<int> send_offsets(size + 1, 0);
  for (int r = 0; r < size; ++r) {
    send_offsets[r + 1] = send_offsets[r] + send_counts[r];
  }
  std::vector<GO> send(static_cast<size_t>(send_offsets[size]));
  std::vector<size_t> position_of_vertex(gids.size());
  {
    auto next = send_offsets;
    for (size_t v = 0; v < gids.size(); ++v) {
      const auto r = static_cast<size_t>(gids[v] % size);
      position_of_vertex[v] = static_cast<size_t>(next[r]) / 3;
      send[next[r]++] = gids[v];
      send[next[r]++] = owned_indices[v] >= 0 ? context.rank : -1;
      send[next[r]++] = owned_indices[v];
    }
  }
  std::vector<int> recv_counts(size, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               context.comm);
  std::vector<int> recv_offsets(size + 1, 0);
  for (int r = 0; r < size; ++r) {
    recv_offsets[r + 1] = recv_offsets[r] + recv_counts[r];
  }
  std::vector<GO> recv(static_cast<size_t>(recv_offsets[size]));
  MPI_Alltoallv(send.data(), send_counts.data(), send_offsets.data(),
                MPI_INT64_T, recv.data(), recv_counts.data(),
                recv_offsets.data(), MPI_INT64_T, context.comm);
  std::map<GO, std::pair<GO, GO>> owner_of;
  for (size_t e = 0; e < recv.size(); e += 3) {
    if (recv[e + 1] < 0) {
      continue;
    }
    if (!owner_of.emplace(recv[e], std::make_pair(recv[e + 1], recv[e + 2]))
           .second) {
      throw pcms_error("Setup: the global id " + std::to_string(recv[e]) +
                       " is owned by two application ranks");
    }
  }
  // answer each entry with its owner, in place
  for (size_t e = 0; e < recv.size(); e += 3) {
    const auto it = owner_of.find(recv[e]);
    if (it == owner_of.end()) {
      throw pcms_error("Setup: no application rank owns the global id " +
                       std::to_string(recv[e]));
    }
    recv[e + 1] = it->second.first;
    recv[e + 2] = it->second.second;
  }
  MPI_Alltoallv(recv.data(), recv_counts.data(), recv_offsets.data(),
                MPI_INT64_T, send.data(), send_counts.data(),
                send_offsets.data(), MPI_INT64_T, context.comm);
  VertexOwners result;
  result.ranks.resize(gids.size());
  result.indices.resize(gids.size());
  for (size_t v = 0; v < gids.size(); ++v) {
    result.ranks[v] = send[3 * position_of_vertex[v] + 1];
    result.indices[v] = send[3 * position_of_vertex[v] + 2];
  }
  return result;
}

template <typename T>
std::vector<GO> GlobalIdsAs(Omega_h::Mesh& mesh, const std::string& name)
{
  const Omega_h::HostRead<T> ids(mesh.get_array<T>(0, name));
  std::vector<GO> gids(static_cast<size_t>(ids.size()));
  for (LO v = 0; v < ids.size(); ++v) {
    gids[static_cast<size_t>(v)] = ids[v];
  }
  return gids;
}

std::vector<GO> VertexGlobalIds(Omega_h::Mesh& mesh, const std::string& name)
{
  if (!mesh.has_tag(0, name)) {
    throw pcms_error("Setup: the mesh has no vertex global id tag '" + name +
                     "'");
  }
  const auto* tag = mesh.get_tagbase(0, name);
  if (Omega_h::is<Omega_h::GO>(tag)) {
    return GlobalIdsAs<Omega_h::GO>(mesh, name);
  }
  return GlobalIdsAs<Omega_h::LO>(mesh, name);
}

/// Whether the mesh's ranks are the application's ranks, so its remotes name
/// application ranks.
bool MeshSpansApplication(Omega_h::Mesh& mesh, const SetupContext& context)
{
  int result = MPI_UNEQUAL;
  MPI_Comm_compare(mesh.comm()->get_impl(), context.comm, &result);
  return result == MPI_IDENT || result == MPI_CONGRUENT;
}

template <typename T>
std::vector<T> HostVector(Omega_h::Read<T> array)
{
  const Omega_h::HostRead<T> host(array);
  std::vector<T> result(static_cast<size_t>(host.size()));
  for (LO i = 0; i < host.size(); ++i) {
    result[static_cast<size_t>(i)] = host[i];
  }
  return result;
}

std::vector<SetupBlock> DescribeMesh(const OmegaHGeometry& geometry,
                                     const std::string& prefix,
                                     Description& description,
                                     const SetupContext& context)
{
  Omega_h::Mesh& mesh = geometry.GetMesh();
  if (mesh.family() != OMEGA_H_SIMPLEX) {
    throw pcms_error("Setup: only Omega_h simplex meshes can be described");
  }
  utility::EnsureClassification(mesh);
  const int dim = mesh.dim();
  description[prefix + ".dimension"] = std::to_string(dim);
  DescribeChart(*geometry.GetCoordinateSystem(), prefix + ".chart",
                description);

  // vertex ownership as the order-1 layouts on the geometry see it
  const auto mesh_owned = HostVector(mesh.owned(0));
  const auto mask = geometry.GetOwnedMask();
  const auto mask_host =
    mask.size() > 0 ? HostVector(mask) : std::vector<Omega_h::I8>{};
  std::vector<bool> owned(mesh_owned.size());
  for (size_t v = 0; v < owned.size(); ++v) {
    owned[v] = mesh_owned[v] != 0 && (mask_host.empty() || mask_host[v] != 0);
  }
  const auto owned_indices = OwnedIndices(owned);
  const auto owners =
    mask.size() == 0 && MeshSpansApplication(mesh, context)
      ? OwnersFromRemotes(mesh, owned_indices)
      : OwnersFromGlobalIds(VertexGlobalIds(mesh, geometry.GetGlobalIdName()),
                            owned_indices, context);
  const auto num_owned_vertices =
    static_cast<GO>(std::count(owned.begin(), owned.end(), true));

  const auto coords = HostVector(mesh.coords());
  const auto vertex_class_dims =
    HostVector(mesh.get_array<Omega_h::I8>(0, "class_dim"));
  const auto vertex_class_ids =
    HostVector(mesh.get_array<Omega_h::ClassId>(0, "class_id"));
  const auto cell_owned = HostVector(mesh.owned(dim));
  const auto cell_class_dims =
    HostVector(mesh.get_array<Omega_h::I8>(dim, "class_dim"));
  const auto cell_class_ids =
    HostVector(mesh.get_array<Omega_h::ClassId>(dim, "class_id"));
  const auto cell_vertices = HostVector(mesh.ask_elem_verts());
  const int verts_per_cell = dim + 1;

  // the owned cells each server rank gets, routed by centroid and
  // classification
  const auto partition = RoutingPartition(context.partition, dim);
  std::vector<std::vector<LO>> cells_of(static_cast<size_t>(context.peer_size));
  std::vector<LO> owned_cell_index(cell_owned.size(), -1);
  LO num_owned_cells = 0;
  for (size_t c = 0; c < cell_owned.size(); ++c) {
    if (cell_owned[c] == 0) {
      continue;
    }
    owned_cell_index[c] = num_owned_cells++;
    std::array<Real, 3> centroid{};
    for (int j = 0; j < verts_per_cell; ++j) {
      const auto v = static_cast<size_t>(cell_vertices[c * verts_per_cell + j]);
      for (int d = 0; d < dim; ++d) {
        centroid[d] += coords[v * dim + d] / verts_per_cell;
      }
    }
    const int rank = RankOf(partition, cell_class_dims[c], cell_class_ids[c],
                            centroid.data(), dim);
    cells_of.at(static_cast<size_t>(rank)).push_back(static_cast<LO>(c));
  }

  std::vector<SetupBlock> blocks(static_cast<size_t>(context.peer_size));
  std::vector<LO> block_vertex(owned.size(), -1);
  for (size_t s = 0; s < blocks.size(); ++s) {
    auto& ints = blocks[s].ints;
    auto& reals = blocks[s].reals;
    std::vector<LO> vertices;
    std::vector<GO> cell_ints;
    for (const LO c : cells_of[s]) {
      cell_ints.push_back(owned_cell_index[c]);
      cell_ints.push_back(cell_class_dims[c]);
      cell_ints.push_back(cell_class_ids[c]);
      for (int j = 0; j < verts_per_cell; ++j) {
        const LO v = cell_vertices[c * verts_per_cell + j];
        if (block_vertex[v] < 0) {
          block_vertex[v] = static_cast<LO>(vertices.size());
          vertices.push_back(v);
        }
        cell_ints.push_back(block_vertex[v]);
      }
    }
    ints = {num_owned_vertices, num_owned_cells,
            static_cast<GO>(cells_of[s].size()),
            static_cast<GO>(vertices.size())};
    ints.insert(ints.end(), cell_ints.begin(), cell_ints.end());
    for (const LO v : vertices) {
      ints.push_back(owners.ranks[v]);
      ints.push_back(owners.indices[v]);
      ints.push_back(vertex_class_dims[v]);
      ints.push_back(vertex_class_ids[v]);
      for (int d = 0; d < dim; ++d) {
        reals.push_back(coords[static_cast<size_t>(v) * dim + d]);
      }
      block_vertex[v] = -1;
    }
  }
  return blocks;
}

RebuiltGeometry RebuildMesh(const Description& description,
                            const std::string& prefix,
                            std::span<const ReceivedBlock> blocks,
                            const SetupContext& context)
{
  const auto dim =
    static_cast<int>(ParseInteger(Require(description, prefix + ".dimension")));
  const int verts_per_cell = dim + 1;
  // the server's ids: the owners' offsets (prefix sums of the application
  // ranks' owned counts) plus the owned index
  std::vector<GO> vertex_offsets(blocks.size() + 1, 0);
  std::vector<GO> cell_offsets(blocks.size() + 1, 0);
  for (size_t r = 0; r < blocks.size(); ++r) {
    if (blocks[r].ints.size() < HeaderSize) {
      throw pcms_error("Setup: malformed mesh block");
    }
    vertex_offsets[r + 1] = vertex_offsets[r] + blocks[r].ints[0];
    cell_offsets[r + 1] = cell_offsets[r] + blocks[r].ints[1];
  }

  struct Vertex
  {
    std::array<Real, 3> coords;
    Omega_h::I8 class_dim;
    Omega_h::ClassId class_id;
  };
  std::map<GO, Vertex> vertices;
  std::vector<GO> cell_globals;
  std::vector<GO> cell_vertex_globals;
  std::vector<Omega_h::I8> cell_class_dims;
  std::vector<Omega_h::ClassId> cell_class_ids;
  for (size_t r = 0; r < blocks.size(); ++r) {
    const auto& ints = blocks[r].ints;
    const auto& reals = blocks[r].reals;
    const auto num_cells = static_cast<size_t>(ints[2]);
    const auto num_vertices = static_cast<size_t>(ints[3]);
    const size_t cell_size = 3 + verts_per_cell;
    const size_t vertex_start = HeaderSize + num_cells * cell_size;
    if (ints.size() != vertex_start + 4 * num_vertices ||
        reals.size() != num_vertices * dim) {
      throw pcms_error("Setup: malformed mesh block");
    }
    std::vector<GO> block_globals(num_vertices);
    for (size_t v = 0; v < num_vertices; ++v) {
      const auto* entry = &ints[vertex_start + 4 * v];
      const auto owner = static_cast<size_t>(entry[0]);
      if (owner >= blocks.size() || entry[1] < 0 ||
          vertex_offsets[owner] + entry[1] >= vertex_offsets[owner + 1]) {
        throw pcms_error("Setup: a vertex has an invalid owner");
      }
      block_globals[v] = vertex_offsets[owner] + entry[1];
      Vertex vertex{};
      for (int d = 0; d < dim; ++d) {
        vertex.coords[d] = reals[v * dim + d];
      }
      vertex.class_dim = static_cast<Omega_h::I8>(entry[2]);
      vertex.class_id = static_cast<Omega_h::ClassId>(entry[3]);
      vertices.emplace(block_globals[v], vertex);
    }
    for (size_t c = 0; c < num_cells; ++c) {
      const auto* entry = &ints[HeaderSize + c * cell_size];
      cell_globals.push_back(cell_offsets[r] + entry[0]);
      cell_class_dims.push_back(static_cast<Omega_h::I8>(entry[1]));
      cell_class_ids.push_back(static_cast<Omega_h::ClassId>(entry[2]));
      for (int j = 0; j < verts_per_cell; ++j) {
        cell_vertex_globals.push_back(
          block_globals.at(static_cast<size_t>(entry[3 + j])));
      }
    }
  }

  // local vertices in the order of their ids, so that the build keeps it
  const auto num_vertices = static_cast<LO>(vertices.size());
  Omega_h::HostWrite<GO> vertex_globals(num_vertices);
  Omega_h::HostWrite<Real> coords(num_vertices * dim);
  Omega_h::HostWrite<Omega_h::I8> vertex_class_dims(num_vertices);
  Omega_h::HostWrite<Omega_h::ClassId> vertex_class_ids(num_vertices);
  std::map<GO, LO> local_of;
  LO next = 0;
  for (const auto& [global, vertex] : vertices) {
    local_of.emplace(global, next);
    vertex_globals[next] = global;
    for (int d = 0; d < dim; ++d) {
      coords[next * dim + d] = vertex.coords[d];
    }
    vertex_class_dims[next] = vertex.class_dim;
    vertex_class_ids[next] = vertex.class_id;
    ++next;
  }
  const auto num_cells = static_cast<LO>(cell_globals.size());
  Omega_h::HostWrite<LO> cell_vertices(num_cells * verts_per_cell);
  for (size_t i = 0; i < cell_vertex_globals.size(); ++i) {
    cell_vertices[static_cast<LO>(i)] = local_of.at(cell_vertex_globals[i]);
  }
  Omega_h::HostWrite<GO> cell_globals_h(num_cells);
  Omega_h::HostWrite<Omega_h::I8> cell_class_dims_h(num_cells);
  Omega_h::HostWrite<Omega_h::ClassId> cell_class_ids_h(num_cells);
  for (LO c = 0; c < num_cells; ++c) {
    cell_globals_h[c] = cell_globals[c];
    cell_class_dims_h[c] = cell_class_dims[c];
    cell_class_ids_h[c] = cell_class_ids[c];
  }

  // the library of the mesh, on the server's ranks; the geometry owns both
  auto library =
    std::make_shared<Omega_h::Library>(nullptr, nullptr, context.comm);
  auto mesh = std::make_shared<Omega_h::Mesh>(library.get());
  mesh->set_comm(library->world());
  mesh->set_parting(OMEGA_H_ELEM_BASED);
  mesh->set_family(OMEGA_H_SIMPLEX);
  mesh->set_dim(dim);
  const Omega_h::GOs globals(vertex_globals.write());
  Omega_h::build_verts_from_globals(mesh.get(), globals);
  Omega_h::build_ents_from_elems2verts(
    mesh.get(), Omega_h::LOs(cell_vertices.write()), globals,
    Omega_h::GOs(cell_globals_h.write()));
  mesh->add_coords(Omega_h::Reals(coords.write()));
  mesh->add_tag<Omega_h::I8>(
    0, "class_dim", 1, Omega_h::Read<Omega_h::I8>(vertex_class_dims.write()));
  mesh->add_tag<Omega_h::ClassId>(
    0, "class_id", 1,
    Omega_h::Read<Omega_h::ClassId>(vertex_class_ids.write()));
  mesh->add_tag<Omega_h::I8>(
    dim, "class_dim", 1, Omega_h::Read<Omega_h::I8>(cell_class_dims_h.write()));
  mesh->add_tag<Omega_h::ClassId>(
    dim, "class_id", 1,
    Omega_h::Read<Omega_h::ClassId>(cell_class_ids_h.write()));

  auto geometry = omega_h::CreateGeometry(
    std::move(mesh), std::move(library),
    ChartFromDescription(description, prefix + ".chart"));
  auto claims = [geometry, vertex_offsets](const FieldLayout& layout) {
    PCMS_ALWAYS_ASSERT(layout.GetGeometry() == geometry &&
                       layout.GetDOFHolderEntityDim() == 0);
    // the holders are the owned vertices; decode their ids into owners
    const auto gids = layout.GetGidsHost();
    std::vector<Claim> result;
    result.reserve(gids.size());
    for (size_t k = 0; k < gids.size(); ++k) {
      const auto it =
        std::upper_bound(vertex_offsets.begin(), vertex_offsets.end(), gids[k]);
      const auto owner = static_cast<LO>(it - vertex_offsets.begin()) - 1;
      result.push_back({static_cast<LO>(k), owner,
                        static_cast<LO>(gids[k] - vertex_offsets[owner])});
    }
    return result;
  };
  return {std::move(geometry), std::move(claims)};
}

class OmegaHKind final : public SpaceKind
{
public:
  [[nodiscard]] std::string_view Name() const noexcept override
  {
    return "omega_h";
  }

  [[nodiscard]] bool Describes(const FunctionSpace& space) const override
  {
    const auto& layout = *space.GetLayout();
    return dynamic_cast<const OmegaHGeometry*>(space.GetGeometry().get()) !=
             nullptr &&
           layout.GetGeometry() == space.GetGeometry() &&
           layout.GetDOFHolderEntityDim() == 0;
  }

  [[nodiscard]] std::vector<SetupBlock> DescribeGeometry(
    const Geometry& geometry, const std::string& prefix,
    Description& description, const SetupContext& context) const override
  {
    return DescribeMesh(dynamic_cast<const OmegaHGeometry&>(geometry), prefix,
                        description, context);
  }

  void DescribeSpace(const FunctionSpace& space, const std::string& prefix,
                     Description& description) const override
  {
    description[prefix + ".order"] = "1";
    description[prefix + ".components"] =
      std::to_string(space.GetLayout()->GetNumComponents());
  }

  [[nodiscard]] RebuiltGeometry RebuildGeometry(
    const Description& description, const std::string& prefix,
    std::span<const ReceivedBlock> blocks,
    const SetupContext& context) const override
  {
    return RebuildMesh(description, prefix, blocks, context);
  }

  [[nodiscard]] std::shared_ptr<const FunctionSpace> RebuildSpace(
    const RebuiltGeometry& geometry, const Description& description,
    const std::string& prefix) const override
  {
    return CreateLagrangeSpace(geometry.geometry, 1,
                               static_cast<int>(ParseInteger(Require(
                                 description, prefix + ".components"))));
  }
};

} // namespace

const SpaceKind& OmegaHSpaceKind()
{
  static const OmegaHKind kind;
  return kind;
}

} // namespace pcms::detail
