#include "description.hpp"
#include "partition.hpp"
#include "pcms/backends/omega_h/omega_h.hpp"
#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/utility/mesh_geometry.hpp"
#include <Omega_h_array_ops.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_dist.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_remotes.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <map>

namespace pcms::detail
{

namespace
{

template <typename T>
Omega_h::Read<T> ToRead(const std::vector<T>& values)
{
  Omega_h::HostWrite<T> host(static_cast<LO>(values.size()));
  std::copy(values.begin(), values.end(), host.begin());
  return host.write();
}

/// The server id of each vertex: its owner's first id plus its index among
/// the owner's owned vertices. Vertex owners come from the mesh's remotes when
/// the mesh spans `app_comm` and has no owned mask, else from the global ids.
Omega_h::HostRead<GO> ServerIds(const omega_h::OmegaHGeometry& geometry,
                                MPI_Comm app_comm)
{
  Omega_h::Mesh& mesh = geometry.GetMesh();
  const auto mask = geometry.GetOwnedMask();
  int same = MPI_UNEQUAL;
  MPI_Comm_compare(mesh.comm()->get_impl(), app_comm, &same);
  Omega_h::CommPtr comm = mesh.comm();
  Omega_h::Remotes owners;
  if (!mask.exists() && (same == MPI_IDENT || same == MPI_CONGRUENT)) {
    owners = mesh.ask_owners(0);
  } else {
    MPI_Comm dup = MPI_COMM_NULL;
    MPI_Comm_dup(app_comm, &dup);
    comm = std::make_shared<Omega_h::Comm>(mesh.library(), dup);
    owners = Omega_h::owners_from_globals(
      comm, mesh.get_array<GO>(0, geometry.GetGlobalIdName()), {});
  }
  const Omega_h::HostRead<Omega_h::I8> owned(
    mask.exists() ? Omega_h::land_each(mesh.owned(0), mask) : mesh.owned(0));
  GO next =
    comm->exscan(GO(std::count(owned.begin(), owned.end(), 1)), OMEGA_H_SUM);
  std::vector<GO> ids(static_cast<size_t>(owned.size()), -1);
  for (LO v = 0; v < owned.size(); ++v) {
    if (owned[v] != 0) {
      ids[v] = next++;
    }
  }
  const Omega_h::Dist to_owners(comm, owners, owned.size());
  return Omega_h::HostRead<GO>(to_owners.invert().exch(
    to_owners.exch_reduce(ToRead(ids), 1, OMEGA_H_MAX), 1));
}

} // namespace

std::optional<std::vector<std::vector<GO>>> DescribeOmegaH(
  const Geometry& geometry, const std::string& prefix, Description& description,
  MPI_Comm comm, redev::Partition partition, size_t num_servers)
{
  const auto* mesh_geometry =
    dynamic_cast<const omega_h::OmegaHGeometry*>(&geometry);
  if (mesh_geometry == nullptr) {
    return std::nullopt;
  }
  Omega_h::Mesh& mesh = mesh_geometry->GetMesh();
  if (mesh.family() != OMEGA_H_SIMPLEX) {
    throw pcms_error("Setup: only Omega_h simplex meshes can be described");
  }
  utility::EnsureClassification(mesh);
  const int dim = mesh.dim();
  description[prefix + ".kind"] = "omega_h";
  description[prefix + ".dimension"] = std::to_string(dim);
  DescribeChart(*geometry.GetCoordinateSystem(), prefix + ".chart",
                description);
  const auto ids = ServerIds(*mesh_geometry, comm);
  if (std::find(ids.begin(), ids.end(), -1) != ids.end()) {
    throw pcms_error("Setup: a mesh vertex has no owner");
  }
  // redev clients receive an RCB partition without its dimension
  if (const auto* rcb = std::get_if<redev::RCBPtn>(&partition)) {
    auto ranks = rcb->GetRanks();
    auto cuts = rcb->GetCuts();
    partition.emplace<redev::RCBPtn>(dim, ranks, cuts);
  }

  using Omega_h::ClassId;
  using Omega_h::I8;
  const Omega_h::HostRead<Real> coords(mesh.coords());
  const Omega_h::HostRead<LO> cell_vertices(mesh.ask_elem_verts());
  const Omega_h::HostRead<I8> cell_owned(mesh.owned(dim));
  const Omega_h::HostRead<I8> vertex_dims(mesh.get_array<I8>(0, "class_dim"));
  const Omega_h::HostRead<ClassId> vertex_ids(
    mesh.get_array<ClassId>(0, "class_id"));
  const Omega_h::HostRead<I8> cell_dims(mesh.get_array<I8>(dim, "class_dim"));
  const Omega_h::HostRead<ClassId> cell_ids(
    mesh.get_array<ClassId>(dim, "class_id"));
  // per server: the owned cells it gets [class dim, class id, vertices by
  // their index in the block], then those vertices [server id, class dim,
  // class id, coordinates as bits]
  std::vector<std::vector<GO>> cells(num_servers);
  std::vector<std::vector<LO>> vertices(num_servers);
  std::vector<std::map<LO, LO>> slots(num_servers);
  for (LO c = 0; c < mesh.nelems(); ++c) {
    if (cell_owned[c] == 0) {
      continue;
    }
    const auto* cell = &cell_vertices[c * (dim + 1)];
    std::array<Real, 3> centroid{};
    for (int j = 0; j <= dim; ++j) {
      for (int d = 0; d < dim; ++d) {
        centroid[d] += coords[cell[j] * dim + d] / (dim + 1);
      }
    }
    const auto s = static_cast<size_t>(std::visit(
      coupler::GetRank{cell_ids[c], cell_dims[c], centroid}, partition));
    cells[s].insert(cells[s].end(), {cell_dims[c], cell_ids[c]});
    for (int j = 0; j <= dim; ++j) {
      const auto [slot, added] =
        slots[s].emplace(cell[j], static_cast<LO>(vertices[s].size()));
      if (added) {
        vertices[s].push_back(cell[j]);
      }
      cells[s].push_back(slot->second);
    }
  }
  std::vector<std::vector<GO>> blocks(num_servers);
  for (size_t s = 0; s < num_servers; ++s) {
    auto& block = blocks[s];
    block.push_back(static_cast<GO>(cells[s].size()) / (dim + 3));
    block.insert(block.end(), cells[s].begin(), cells[s].end());
    for (const LO v : vertices[s]) {
      block.insert(block.end(), {ids[v], vertex_dims[v], vertex_ids[v]});
      for (int d = 0; d < dim; ++d) {
        block.push_back(std::bit_cast<GO>(coords[v * dim + d]));
      }
    }
  }
  return blocks;
}

std::shared_ptr<const Geometry> RebuildOmegaH(
  const Description& description, const std::string& prefix,
  const std::vector<std::vector<GO>>& blocks, MPI_Comm comm)
{
  const auto dim =
    static_cast<int>(ParseInteger(Require(description, prefix + ".dimension")));
  const auto record = static_cast<size_t>(dim) + 3;
  const auto first_vertex = [record](const std::vector<GO>& block) {
    return 1 + static_cast<size_t>(block[0]) * record;
  };
  // local vertices in the order of their server ids
  std::map<GO, LO> local;
  for (const auto& block : blocks) {
    for (auto k = first_vertex(block); k < block.size(); k += record) {
      local.emplace(block[k], 0);
    }
  }
  std::vector<GO> globals;
  for (auto& [id, index] : local) {
    index = static_cast<LO>(globals.size());
    globals.push_back(id);
  }
  std::vector<Real> coords(globals.size() * dim);
  std::vector<Omega_h::I8> vertex_dims(globals.size());
  std::vector<Omega_h::ClassId> vertex_ids(globals.size());
  std::vector<LO> cell_vertices;
  std::vector<Omega_h::I8> cell_dims;
  std::vector<Omega_h::ClassId> cell_ids;
  for (const auto& block : blocks) {
    const auto vertices = first_vertex(block);
    for (auto k = vertices; k < block.size(); k += record) {
      const LO v = local[block[k]];
      vertex_dims[v] = static_cast<Omega_h::I8>(block[k + 1]);
      vertex_ids[v] = static_cast<Omega_h::ClassId>(block[k + 2]);
      for (int d = 0; d < dim; ++d) {
        coords[v * dim + d] = std::bit_cast<Real>(block[k + 3 + d]);
      }
    }
    for (size_t c = 1; c < vertices; c += record) {
      cell_dims.push_back(static_cast<Omega_h::I8>(block[c]));
      cell_ids.push_back(static_cast<Omega_h::ClassId>(block[c + 1]));
      for (int j = 0; j <= dim; ++j) {
        cell_vertices.push_back(
          local[block[vertices + record * block[c + 2 + j]]]);
      }
    }
  }
  // the geometry owns the mesh and its library on the server's ranks
  auto library = std::make_shared<Omega_h::Library>(nullptr, nullptr, comm);
  auto mesh = std::make_shared<Omega_h::Mesh>(library.get());
  Omega_h::build_from_elems2verts(mesh.get(), library->world(), OMEGA_H_SIMPLEX,
                                  dim, ToRead(cell_vertices), ToRead(globals));
  mesh->add_coords(ToRead(coords));
  mesh->add_tag(0, "class_dim", 1, ToRead(vertex_dims));
  mesh->add_tag(0, "class_id", 1, ToRead(vertex_ids));
  mesh->add_tag(dim, "class_dim", 1, ToRead(cell_dims));
  mesh->add_tag(dim, "class_id", 1, ToRead(cell_ids));
  return omega_h::CreateGeometry(
    std::move(mesh), std::move(library),
    ChartFromDescription(description, prefix + ".chart"));
}

} // namespace pcms::detail
