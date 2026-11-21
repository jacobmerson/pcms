// Functions described by a client and rebuilt by the server in
// ApplicationComm::Setup: the server builds no space of its own for them.
//
// usage: test_coupler_setup <server|client> <BP4|SST> <remotes|gids>
//
// The client describes a P1 and a P0 function on one uniform grid and a P1
// function on an Omega_h mesh. With "remotes" the mesh is distributed over the
// client's ranks and identified by Omega_h's remotes; with "gids" every
// client rank holds a serial piece of it with global vertex ids and an owned
// mask. The server checks the received values exactly, interpolates the mesh
// function onto a space of its own and sends back twice the values, which
// the client checks.

#include "pcms/backends/omega_h/omega_h.hpp"
#include "pcms/backends/uniform_grid/geometry.hpp"
#include "pcms/coupler/coupler.hpp"
#include "pcms/field/function_spaces.hpp"
#include "pcms/transfer/interpolator.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace
{

using pcms::LO;
using pcms::Real;

[[noreturn]] void Fail(const std::string& message)
{
  std::cerr << "FAILED: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void Require(bool condition, const std::string& message)
{
  if (!condition) {
    Fail(message);
  }
}

Real F(Real x, Real y)
{
  return 1.0 + 2.0 * x + 3.0 * y;
}

/// The owned DOF holders' coordinates, on the host.
std::vector<std::array<Real, 2>> HolderCoords(const pcms::FieldLayout& layout)
{
  const auto coords = layout.GetDOFHolderCoordinates().GetValues();
  Kokkos::View<Real**, pcms::DeviceMemorySpace> staged(
    "coords", coords.extent(0), coords.extent(1));
  Kokkos::parallel_for(
    Kokkos::RangePolicy<pcms::DeviceMemorySpace::execution_space>(
      0, coords.extent(0)),
    KOKKOS_LAMBDA(int i) {
      staged(i, 0) = coords(i, 0);
      staged(i, 1) = coords(i, 1);
    });
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), staged);
  std::vector<std::array<Real, 2>> result(host.extent(0));
  for (size_t i = 0; i < host.extent(0); ++i) {
    result[i] = {host(i, 0), host(i, 1)};
  }
  return result;
}

std::vector<Real> Values(const pcms::Field<Real>& field)
{
  const auto values = field.GetDOFHolderDataHost().GetValues();
  std::vector<Real> result(values.extent(0));
  for (size_t i = 0; i < values.extent(0); ++i) {
    result[i] = values(i, 0);
  }
  return result;
}

void SetValues(pcms::Field<Real>& field, std::vector<Real> values)
{
  field.SetDOFHolderDataHost(pcms::ValueView<const Real, pcms::HostMemorySpace>(
    pcms::ValueBasis{}, pcms::Rank2View<const Real, pcms::HostMemorySpace>(
                          values.data(), values.size(), 1)));
}

/// f, times `scale`, at the owned DOF holders of the field's layout
std::vector<Real> Expected(const pcms::Field<Real>& field, Real scale)
{
  std::vector<Real> values;
  for (const auto& x : HolderCoords(field.GetLayout())) {
    values.push_back(scale * F(x[0], x[1]));
  }
  return values;
}

std::shared_ptr<const pcms::Geometry> GridGeometry()
{
  pcms::UniformGrid<2> grid{
    .edge_length = {1.0, 1.0}, .bot_left = {0.0, 0.0}, .divisions = {6, 5}};
  return pcms::uniform_grid::CreateGeometry(grid,
                                            pcms::csys::CreateCartesian(2));
}

/// A serial piece of the 8x8 unit box for each rank: the elements whose
/// centroid lies in the rank's slab of x, with the box's vertex ids as global
/// ids, and the owned mask that gives a shared vertex to the lowest rank.
struct Piece
{
  Omega_h::Mesh mesh;
  Omega_h::Read<Omega_h::I8> owned_mask;
};

Piece BuildPiece(Omega_h::Library& lib, int rank, int size)
{
  auto full =
    Omega_h::build_box(lib.self(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 8, 8, 0);
  const auto coords = Omega_h::HostRead<Real>(full.coords());
  const auto elem_verts = Omega_h::HostRead<LO>(full.ask_elem_verts());
  const auto class_dims_v =
    Omega_h::HostRead<Omega_h::I8>(full.get_array<Omega_h::I8>(0, "class_dim"));
  const auto class_ids_v = Omega_h::HostRead<Omega_h::ClassId>(
    full.get_array<Omega_h::ClassId>(0, "class_id"));
  const auto class_dims_e =
    Omega_h::HostRead<Omega_h::I8>(full.get_array<Omega_h::I8>(2, "class_dim"));
  const auto class_ids_e = Omega_h::HostRead<Omega_h::ClassId>(
    full.get_array<Omega_h::ClassId>(2, "class_id"));
  const LO nelems = full.nelems();
  const LO nverts = full.nverts();
  auto piece_of = [&](LO e) {
    Real x = 0;
    for (int j = 0; j < 3; ++j) {
      x += coords[2 * elem_verts[3 * e + j]] / 3;
    }
    return std::min(static_cast<int>(x * size), size - 1);
  };
  // the lowest piece around each vertex owns it
  std::vector<int> owner(nverts, size);
  for (LO e = 0; e < nelems; ++e) {
    for (int j = 0; j < 3; ++j) {
      auto& o = owner[elem_verts[3 * e + j]];
      o = std::min(o, piece_of(e));
    }
  }
  std::vector<LO> local(nverts, -1);
  std::vector<LO> globals;
  std::vector<LO> piece_elem_verts;
  std::vector<LO> piece_elems;
  for (LO e = 0; e < nelems; ++e) {
    if (piece_of(e) != rank) {
      continue;
    }
    piece_elems.push_back(e);
    for (int j = 0; j < 3; ++j) {
      const LO v = elem_verts[3 * e + j];
      if (local[v] < 0) {
        local[v] = static_cast<LO>(globals.size());
        globals.push_back(v);
      }
      piece_elem_verts.push_back(local[v]);
    }
  }
  const auto n = static_cast<LO>(globals.size());
  Omega_h::HostWrite<Real> piece_coords(2 * n);
  Omega_h::HostWrite<Omega_h::GO> piece_globals(n);
  Omega_h::HostWrite<Omega_h::I8> mask(n);
  Omega_h::HostWrite<Omega_h::I8> vdims(n);
  Omega_h::HostWrite<Omega_h::ClassId> vids(n);
  for (LO v = 0; v < n; ++v) {
    const LO g = globals[v];
    piece_coords[2 * v] = coords[2 * g];
    piece_coords[2 * v + 1] = coords[2 * g + 1];
    piece_globals[v] = g;
    mask[v] = owner[g] == rank ? 1 : 0;
    vdims[v] = class_dims_v[g];
    vids[v] = class_ids_v[g];
  }
  const auto ne = static_cast<LO>(piece_elems.size());
  Omega_h::HostWrite<LO> ev2v(3 * ne);
  Omega_h::HostWrite<Omega_h::I8> edims(ne);
  Omega_h::HostWrite<Omega_h::ClassId> eids(ne);
  for (LO e = 0; e < ne; ++e) {
    for (int j = 0; j < 3; ++j) {
      ev2v[3 * e + j] = piece_elem_verts[3 * e + j];
    }
    edims[e] = class_dims_e[piece_elems[e]];
    eids[e] = class_ids_e[piece_elems[e]];
  }
  Piece piece{Omega_h::Mesh(&lib), Omega_h::Read<Omega_h::I8>(mask.write())};
  Omega_h::build_from_elems_and_coords(&piece.mesh, OMEGA_H_SIMPLEX, 2,
                                       Omega_h::LOs(ev2v.write()),
                                       Omega_h::Reals(piece_coords.write()));
  piece.mesh.set_tag(0, "global",
                     Omega_h::Read<Omega_h::GO>(piece_globals.write()));
  piece.mesh.add_tag(0, "class_dim", 1,
                     Omega_h::Read<Omega_h::I8>(vdims.write()));
  piece.mesh.add_tag(0, "class_id", 1,
                     Omega_h::Read<Omega_h::ClassId>(vids.write()));
  piece.mesh.add_tag(2, "class_dim", 1,
                     Omega_h::Read<Omega_h::I8>(edims.write()));
  piece.mesh.add_tag(2, "class_id", 1,
                     Omega_h::Read<Omega_h::ClassId>(eids.write()));
  return piece;
}

void Client(Omega_h::Library& lib, redev::TransportType transport,
            const adios2::Params& params, bool gids)
{
  auto world = lib.world();
  const int rank = world->rank();
  // one grid geometry with two spaces on it
  const auto grid = GridGeometry();
  auto grid_p1 =
    pcms::CreateLagrangeSpace(grid, 1, 1)->CreateFunction<Real>("grid_p1");
  auto grid_p0 =
    pcms::CreateLagrangeSpace(grid, 0, 1)->CreateFunction<Real>("grid_p0");

  // the mesh: distributed, or a serial piece per rank with global ids
  Omega_h::Mesh distributed(&lib);
  Piece piece{Omega_h::Mesh(&lib), {}};
  std::shared_ptr<const pcms::Geometry> mesh_geometry;
  pcms::LagrangeOptions options;
  if (gids) {
    piece = BuildPiece(lib, rank, world->size());
    mesh_geometry = pcms::omega_h::CreateGeometry(
      piece.mesh, pcms::csys::CreateCartesian(2),
      pcms::omega_h::OmegaHGeometryOptions{.owned_mask = piece.owned_mask});
    // the owned mask is supported by the Omega_h backend
    options.backend = pcms::LagrangeBackend::OmegaH;
  } else {
    distributed =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 8, 8, 0);
    mesh_geometry = pcms::omega_h::CreateGeometry(
      distributed, pcms::csys::CreateCartesian(2));
  }
  auto mesh_p1 = pcms::CreateLagrangeSpace(mesh_geometry, 1, 1, options)
                   ->CreateFunction<Real>("mesh_p1");

  // the meshes outlive the coupler, which holds the functions
  pcms::CouplerComm coupler("setup_client", world->get_impl(), false, {});
  auto* app = coupler.AddApplication("setup_app", "", transport, params);

  SetValues(grid_p1, Expected(grid_p1, 1.0));
  SetValues(grid_p0, Expected(grid_p0, 1.0));
  SetValues(mesh_p1, Expected(mesh_p1, 1.0));
  auto h_grid_p1 = app->DescribeFunction(std::move(grid_p1));
  auto h_grid_p0 = app->DescribeFunction(std::move(grid_p0));
  auto h_mesh_p1 = app->DescribeFunction(std::move(mesh_p1));
  app->Setup();

  app->SendPhase([&]() {
    h_grid_p1.Send();
    h_grid_p0.Send();
    h_mesh_p1.Send();
  });
  app->ReceivePhase([&]() {
    h_grid_p1.Receive();
    h_grid_p0.Receive();
    h_mesh_p1.Receive();
  });

  // the server doubled the values; a replicated grid is sent and received by
  // the first rank only
  const Real grid_scale = rank == 0 ? 2.0 : 1.0;
  Require(Values(h_grid_p1.GetField()) ==
            Expected(h_grid_p1.GetField(), grid_scale),
          "client: grid P1 values after the round trip");
  Require(Values(h_grid_p0.GetField()) ==
            Expected(h_grid_p0.GetField(), grid_scale),
          "client: grid P0 values after the round trip");
  Require(Values(h_mesh_p1.GetField()) == Expected(h_mesh_p1.GetField(), 2.0),
          "client: mesh P1 values after the round trip");
  std::cerr << "client rank " << rank << " passed\n";
}

/// The number of holders whose value is f; the others must be 0 (not sent to
/// this server rank).
LO CountExact(const pcms::Field<Real>& field, const std::string& name)
{
  const auto values = Values(field);
  const auto expected = Expected(field, 1.0);
  LO exact = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] == expected[i]) {
      ++exact;
    } else {
      Require(values[i] == 0.0, "server: " + name + " holder " +
                                  std::to_string(i) + " has a wrong value");
    }
  }
  return exact;
}

LO Sum(LO value, MPI_Comm comm)
{
  LO sum = 0;
  MPI_Allreduce(&value, &sum, 1, MPI_INT, MPI_SUM, comm);
  return sum;
}

void Server(Omega_h::Library& lib, redev::TransportType transport,
            const adios2::Params& params)
{
  auto world = lib.world();
  const MPI_Comm comm = world->get_impl();
  const int size = world->size();
  // the server's partition, fixed in advance: halves of x for two ranks
  std::vector<int> ranks(size);
  std::iota(ranks.begin(), ranks.end(), 0);
  std::vector<Real> cuts =
    size == 1 ? std::vector<Real>{0.0} : std::vector<Real>{0.0, 0.5};
  Require(size <= 2, "the server runs on one or two ranks");
  pcms::CouplerComm coupler("setup_server", comm, true,
                            redev::Partition{redev::RCBPtn{2, ranks, cuts}});
  auto* app = coupler.AddApplication("setup_app", "", transport, params);
  app->Expect({"grid_p1", "grid_p0", "mesh_p1"});
  app->Setup();

  // no space was built here: the functions are the application's
  auto grid_p1 = app->GetFunction<Real>("grid_p1");
  auto grid_p0 = app->GetFunction<Real>("grid_p0");
  auto mesh_p1 = app->GetFunction<Real>("mesh_p1");
  Require(grid_p1.GetSpace().GetGeometry() == grid_p0.GetSpace().GetGeometry(),
          "server: the two grid spaces share their geometry");
  app->ReceivePhase([&]() {
    grid_p1.Receive();
    grid_p0.Receive();
    mesh_p1.Receive();
  });

  // every grid holder arrives at exactly one server rank, every mesh vertex
  // at its owner
  Require(Sum(CountExact(grid_p1.GetField(), "grid_p1"), comm) == 7 * 6,
          "server: every grid P1 holder arrives once");
  Require(Sum(CountExact(grid_p0.GetField(), "grid_p0"), comm) == 6 * 5,
          "server: every grid P0 holder arrives once");
  const auto& mesh_field = mesh_p1.GetField();
  Require(Values(mesh_field) == Expected(mesh_field, 1.0),
          "server: mesh P1 values");
  Require(Sum(mesh_field.GetLayout().GetNumDofHolder(), comm) == 9 * 9,
          "server: the rebuilt mesh has the box's vertices");

  // a local task: interpolate the received mesh function onto a space of the
  // server's own
  const auto target_space = pcms::CreateLagrangeSpace(
    pcms::uniform_grid::CreateGeometry(
      pcms::UniformGrid<2>{
        .edge_length = {0.8, 0.8}, .bot_left = {0.1, 0.1}, .divisions = {4, 4}},
      pcms::csys::CreateCartesian(2)),
    1, 1);
  auto target = target_space->CreateFunction<Real>("target");
  const Real missing = std::numeric_limits<Real>::quiet_NaN();
  pcms::CreateInterpolator<Real>(
    mesh_p1.GetSpacePtr(), target_space,
    pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, missing})
    ->Apply(mesh_field, target);
  const auto interpolated = Values(target);
  const auto expected = Expected(target, 1.0);
  LO found = 0;
  for (size_t i = 0; i < interpolated.size(); ++i) {
    // points in another server rank's part of the mesh are not found here
    if (std::isnan(interpolated[i])) {
      continue;
    }
    ++found;
    Require(std::abs(interpolated[i] - expected[i]) < 1e-12,
            "server: interpolated value");
  }
  Require(size > 1 || found == static_cast<LO>(interpolated.size()),
          "server: all points are in the mesh");
  Require(found > 0, "server: some points are in this rank's mesh");

  for (auto* handle : {&grid_p1, &grid_p0, &mesh_p1}) {
    auto values = Values(handle->GetField());
    for (auto& value : values) {
      value *= 2;
    }
    SetValues(handle->GetField(), values);
  }
  app->SendPhase([&]() {
    grid_p1.Send();
    grid_p0.Send();
    mesh_p1.Send();
  });
  std::cerr << "server rank " << world->rank() << " passed\n";
}

} // namespace

int main(int argc, char** argv)
{
  Omega_h::Library lib(&argc, &argv);
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <server|client> <BP4|SST> <remotes|gids>\n";
    return EXIT_FAILURE;
  }
  const std::string role = argv[1];
  const std::string engine = argv[2];
  const bool gids = std::string(argv[3]) == "gids";
  const auto transport =
    engine == "SST" ? redev::TransportType::SST : redev::TransportType::BP4;
  // SST: speculative preload off works around ADIOS2 issue 5249
  const adios2::Params params =
    engine == "SST" ? adios2::Params{{"OpenTimeoutSecs", "60"},
                                     {"SpeculativePreloadMode", "OFF"}}
                    : adios2::Params{{"Streaming", "On"},
                                     {"OpenTimeoutSecs", "60"},
                                     {"BeginStepPollingFrequencySecs", "0.05"}};
  try {
    if (role == "server") {
      Server(lib, transport, params);
    } else {
      Client(lib, transport, params, gids);
    }
  } catch (const std::exception& e) {
    Fail(std::string("exception: ") + e.what());
  }
  return EXIT_SUCCESS;
}
