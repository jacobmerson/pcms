#include <algorithm>
#include <array>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <iostream>
#include <Omega_h_mesh.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_for.hpp>
#include <redev.h>
#include <vector>
#include "pcms/coupler/detail/field_communicator.hpp"
#include "pcms/coupler/coupler.hpp"
#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/backends/omega_h/omega_h.hpp"
#include "pcms/field/function_spaces.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/field/detail/simple_field_data.hpp"
#include "test_support.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"

namespace pcms
{
using omega_h::CreateGeometry;
} // namespace pcms

namespace ts = test_support;
using pcms::Real;

ts::ClassificationPartition getClassificationPartition(Omega_h::Mesh& mesh)
{
  ts::ClassificationPartition cp;
  std::map<redev::ClassPtn::ModelEnt, int> ent_to_rank;

  for (int e = 0; e <= 0; ++e) {
    auto ids_d = mesh.get_array<Omega_h::ClassId>(e, "class_id");
    auto dims_d = mesh.get_array<Omega_h::I8>(e, "class_dim");
    auto remotes = mesh.ask_owners(e);

    auto ids = Omega_h::HostRead(ids_d);
    auto dims = Omega_h::HostRead(dims_d);
    auto remotes_ranks = Omega_h::HostRead(remotes.ranks);

    for (int i = 0; i < mesh.nents(e); ++i) {
      redev::ClassPtn::ModelEnt ent({dims[i], ids[i]});
      int rank = remotes_ranks[i];
      auto it = ent_to_rank.find(ent);
      if (it == ent_to_rank.end()) {
        ent_to_rank[ent] = rank;
        cp.ranks.push_back(rank);
        cp.modelEnts.push_back(ent);
      } else {
        PCMS_ALWAYS_ASSERT(rank == it->second);
      }
    }
  }
  return cp;
}

redev::ClassPtn setupServerPartition(Omega_h::Mesh& mesh,
                                     std::string_view cpnFileName)
{
  auto ohComm = mesh.comm();
  const auto facePartition = !ohComm->rank()
                               ? ts::readClassPartitionFile(cpnFileName)
                               : ts::ClassificationPartition();
  ts::migrateMeshElms(mesh, facePartition);
  REDEV_ALWAYS_ASSERT(mesh.nelems()); // all ranks should have elements
  auto ptn = getClassificationPartition(mesh);
  return redev::ClassPtn(MPI_COMM_WORLD, ptn.ranks, ptn.modelEnts);
}

// Test that two fields sharing a layout via AddLayout use the same
// FieldLayoutCommunicator rather than creating separate ones.
static void test_shared_layout(Omega_h::Library& lib,
                               std::string_view mesh_file,
                               std::string_view cpn_file, bool is_server)
{
  auto world = lib.world();
  MPI_Comm mpi_comm = world->get_impl();
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(std::string(mesh_file).c_str(), lib.world(), &mesh);

  if (is_server) {
    auto partition = setupServerPartition(mesh, cpn_file);
    pcms::CouplerComm cpl("shared_layout_server", mpi_comm, true,
                          redev::Partition{partition});
    auto* app = cpl.AddApplication("shared_layout");
    auto factory = pcms::CreateLagrangeSpace(
      pcms::CreateGeometry(mesh, pcms::detail::DeferredCartesian()), 1, 1,
      pcms::LagrangeOptions{.layout_name = "shared"});
    auto f1 = app->AddField(factory->CreateFunction<Real>("field_a"));
    PCMS_ALWAYS_ASSERT(app->GetLayoutCommunicatorCount() == 1);
    auto f2 = app->AddField(factory->CreateFunction<Real>("field_b"));
    PCMS_ALWAYS_ASSERT(app->GetLayoutCommunicatorCount() == 1); // shared layout
    app->ReceivePhase([&]() {
      f1.Receive();
      f2.Receive();
    });
    app->SendPhase([&]() {
      f1.Send();
      f2.Send();
    });
  } else {
    pcms::CouplerComm cpl("shared_layout_client", mpi_comm, false,
                          redev::Partition{});
    auto* app = cpl.AddApplication("shared_layout");
    auto factory = pcms::CreateLagrangeSpace(
      pcms::CreateGeometry(mesh, pcms::detail::DeferredCartesian()), 1, 1,
      pcms::LagrangeOptions{.layout_name = "shared"});
    auto f1 = app->AddField(factory->CreateFunction<Real>("field_a"));
    auto f2 = app->AddField(factory->CreateFunction<Real>("field_b"));
    app->SendPhase([&]() {
      f1.Send();
      f2.Send();
    });
    app->ReceivePhase([&]() {
      f1.Receive();
      f2.Receive();
    });
  }
}

// Move the elements to the ranks whose RCB region holds their centroid.
static void migrateToRcbRegions(Omega_h::Mesh& mesh, redev::RCBPtn& ptn)
{
  auto comm = mesh.comm();
  const auto coords = Omega_h::HostRead(mesh.coords());
  const auto elem_verts = Omega_h::HostRead(mesh.ask_elem_verts());
  const int dim = mesh.dim();
  std::vector<int> dest(static_cast<size_t>(mesh.nelems()));
  for (size_t e = 0; e < dest.size(); ++e) {
    std::array<Real, 3> centroid{};
    for (int v = 0; v <= dim; ++v)
      for (int d = 0; d < dim; ++d)
        centroid[d] +=
          coords[elem_verts[e * (dim + 1) + v] * dim + d] / (dim + 1);
    dest[e] = ptn.GetRank(centroid);
  }
  const int nproc = comm->size();
  const int n = static_cast<int>(dest.size());
  std::vector<int> counts(nproc), displs(nproc, 0);
  MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm->get_impl());
  std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
  std::vector<int> all_dest(displs.back() + counts.back());
  MPI_Allgatherv(dest.data(), n, MPI_INT, all_dest.data(), counts.data(),
                 displs.data(), MPI_INT, comm->get_impl());
  // pull the elements that now belong to this rank
  std::vector<Omega_h::I32> owner_ranks;
  std::vector<Omega_h::LO> owner_idxs;
  for (int r = 0; r < nproc; ++r)
    for (int i = 0; i < counts[r]; ++i)
      if (all_dest[displs[r] + i] == comm->rank()) {
        owner_ranks.push_back(r);
        owner_idxs.push_back(i);
      }
  Omega_h::HostWrite<Omega_h::I32> ranks_h(owner_ranks.size());
  Omega_h::HostWrite<Omega_h::LO> idxs_h(owner_idxs.size());
  std::copy(owner_ranks.begin(), owner_ranks.end(), ranks_h.data());
  std::copy(owner_idxs.begin(), owner_idxs.end(), idxs_h.data());
  Omega_h::Write<Omega_h::I32> ranks_d(ranks_h);
  Omega_h::Write<Omega_h::LO> idxs_d(idxs_h);
  Omega_h::Remotes owners(ranks_d, idxs_d);
  mesh.migrate(owners);
}

// Couple a client and a server that both have several ranks, with an RCB
// partition cut into one region per server rank. The client sends gid + 1 and
// every value must arrive on the server rank whose region holds the vertex.
static void test_rcb_routing(Omega_h::Library& lib, bool is_server)
{
  auto world = lib.world();
  MPI_Comm mpi_comm = world->get_impl();
  auto mesh = Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 4, 4, 0);
  // Vertices on the cut belong to the second region and Omega_h gives them to
  // the lowest rank holding them, so rank 0 owns the second region.
  redev::LOs ptn_ranks = {1, 0};
  redev::Reals ptn_cuts = {0, 0.5}; // heap layout: cuts[1] is the cut
  redev::RCBPtn ptn(2, ptn_ranks, ptn_cuts);
  if (is_server)
    migrateToRcbRegions(mesh, ptn);
  pcms::CouplerComm cpl("rcb_routing", mpi_comm, is_server,
                        is_server ? redev::Partition{ptn} : redev::Partition{});
  auto* app = cpl.AddApplication("rcb_routing");
  auto factory = pcms::CreateLagrangeSpace(
    pcms::CreateGeometry(mesh, pcms::detail::DeferredCartesian()), 1, 1,
    pcms::LagrangeOptions{.layout_name = "rcb_routing"});
  auto layout = factory->GetLayout();
  auto gids = layout->GetGidsHost();
  const auto owned = layout->GetOwnedHost();
  const auto n = layout->GetNumDofHolder();
  Omega_h::HostWrite<Real> values(n);
  for (int i = 0; i < n; ++i)
    values[i] = is_server ? 0 : gids[i] + 1;
  auto field = app->AddField(factory->CreateFunction<Real>("field"));
  field.GetField().SetDOFHolderDataUncheckedHost(
    pcms::Rank2View<const Real, pcms::HostMemorySpace>(values.data(), n, 1));
  if (!is_server) {
    app->SendPhase([&]() { field.Send(); });
    return;
  }
  app->ReceivePhase([&]() { field.Receive(); });
  const auto received = field.GetField().GetDOFHolderDataHost().GetValues();
  const auto coords = Omega_h::HostRead(mesh.coords());
  int num_owned = 0;
  for (int i = 0; i < n; ++i) {
    if (!owned[i])
      continue;
    ++num_owned;
    std::array<Real, 3> pt = {coords[2 * i], coords[2 * i + 1], 0};
    PCMS_ALWAYS_ASSERT(received(i, 0) == gids[i] + 1);
    PCMS_ALWAYS_ASSERT(ptn.GetRank(pt) == world->rank());
  }
  PCMS_ALWAYS_ASSERT(num_owned > 0);
  PCMS_ALWAYS_ASSERT(world->allreduce(num_owned, OMEGA_H_SUM) ==
                     mesh.nglobal_ents(0));
}

void client1(MPI_Comm comm, Omega_h::Mesh& mesh, std::string comm_name,
             int order, const adios2::Params& params)
{
  redev::Redev rdv(MPI_COMM_WORLD);
  auto channel =
    rdv.CreateAdiosChannel("field2_chan1", params, redev::TransportType::BP4);

  auto geometry = pcms::CreateGeometry(mesh, pcms::detail::DeferredCartesian());
  auto factory = pcms::CreateLagrangeSpace(geometry, order, 1);
  auto layout = factory->GetLayout();
  auto gids = layout->GetGidsHost();
  const auto n = layout->GetNumDofHolder();
  Omega_h::HostWrite<Real> ids(n);
  PCMS_ALWAYS_ASSERT(n == gids.size());
  Kokkos::parallel_for(
    "id gid", Kokkos::RangePolicy<pcms::HostMemorySpace::execution_space>(0, n),
    [=](int i) { ids[i] = gids[i]; });

  auto field = factory->CreateFunction<Real>();
  field.SetDOFHolderDataUncheckedHost(
    pcms::Rank2View<const Real, pcms::HostMemorySpace>(ids.data(), n, 1));

  pcms::detail::FieldLayoutCommunicator layout_comm(comm_name + "1", comm, rdv,
                                                    channel, *layout);
  pcms::detail::FieldCommunicator<pcms::Real> field_comm(layout_comm.GetName(),
                                                         layout_comm, field);

  channel.BeginSendCommunicationPhase();
  field_comm.Send();
  channel.EndSendCommunicationPhase();
}

void client2(MPI_Comm comm, Omega_h::Mesh& mesh, std::string comm_name,
             int order, const adios2::Params& params)
{
  redev::Redev rdv(MPI_COMM_WORLD);
  auto channel =
    rdv.CreateAdiosChannel("field2_chan2", params, redev::TransportType::BP4);

  auto geometry = pcms::CreateGeometry(mesh, pcms::detail::DeferredCartesian());
  auto factory = pcms::CreateLagrangeSpace(geometry, order, 1);
  auto layout = factory->GetLayout();
  auto gids = layout->GetGidsHost();
  const auto n = layout->GetNumDofHolder();

  auto field = factory->CreateFunction<Real>();
  pcms::detail::FieldLayoutCommunicator layout_comm(comm_name + "2", comm, rdv,
                                                    channel, *layout);
  pcms::detail::FieldCommunicator<pcms::Real> field_comm(layout_comm.GetName(),
                                                         layout_comm, field);

  channel.BeginReceiveCommunicationPhase();
  field_comm.Receive();
  channel.EndReceiveCommunicationPhase();

  auto copied_array =
    pcms::detail::FlattenToRank1View(field.GetDOFHolderDataHost().GetValues());

  PCMS_ALWAYS_ASSERT(copied_array.size() == gids.size());

  int expected = 0;
  int sum = 0;
  Kokkos::parallel_reduce(
    "reduce1",
    Kokkos::RangePolicy<pcms::HostMemorySpace::execution_space>(0, gids.size()),
    KOKKOS_LAMBDA(int /*i*/, int& local_sum) { local_sum += 1; }, expected);
  Kokkos::parallel_reduce(
    "reduce2",
    Kokkos::RangePolicy<pcms::HostMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(int i, int& local_sum) {
      local_sum += (int)copied_array[i] == gids[i];
    },
    sum);

  if (sum != expected) {
    std::cerr << "Field Communication Failed, expected " << expected << " got "
              << sum << std::endl;
    exit(EXIT_FAILURE);
  } else {
    std::cerr << "Field Communication Passed\n";
  }
}

void server(MPI_Comm comm, Omega_h::Mesh& mesh, std::string comm_name,
            int order, const adios2::Params& params,
            const std::string& cpn_filename)
{
  redev::Redev rdv(MPI_COMM_WORLD,
                   redev::Partition{setupServerPartition(mesh, cpn_filename)},
                   redev::ProcessType::Server);
  auto channel1 =
    rdv.CreateAdiosChannel("field2_chan1", params, redev::TransportType::BP4);
  auto channel2 =
    rdv.CreateAdiosChannel("field2_chan2", params, redev::TransportType::BP4);

  auto geometry = pcms::CreateGeometry(mesh, pcms::detail::DeferredCartesian());
  auto factory = pcms::CreateLagrangeSpace(geometry, order, 1);
  auto layout = factory->GetLayout();
  const auto n = layout->GetNumDofHolder();
  Omega_h::HostWrite<Real> ids(n);
  Kokkos::parallel_for(
    "id 0", Kokkos::RangePolicy<pcms::HostMemorySpace::execution_space>(0, n),
    [=](int i) { ids[i] = 0; });

  auto field = factory->CreateFunction<Real>();
  pcms::detail::FieldLayoutCommunicator layout_comm1(comm_name + "1", comm, rdv,
                                                     channel1, *layout);
  pcms::detail::FieldLayoutCommunicator layout_comm2(comm_name + "2", comm, rdv,
                                                     channel2, *layout);
  pcms::detail::FieldCommunicator<pcms::Real> field_comm1(
    layout_comm1.GetName(), layout_comm1, field);
  pcms::detail::FieldCommunicator<pcms::Real> field_comm2(
    layout_comm2.GetName(), layout_comm2, field);

  channel1.BeginReceiveCommunicationPhase();
  field_comm1.Receive();
  channel1.EndReceiveCommunicationPhase();

  channel2.BeginSendCommunicationPhase();
  field_comm2.Send();
  channel2.EndSendCommunicationPhase();
}

int main(int argc, char** argv)
{
  try {
    auto lib = Omega_h::Library(&argc, &argv);
    auto world = lib.world();
    int rank = world->rank();
    if (argc != 4) {
      std::cerr << "Usage: " << argv[0]
                << " <clientId=-1|0|1|2|3|4|5> /path/to/omega_h/mesh"
                << " /path/to/partitionFile.cpn\n";
      exit(EXIT_FAILURE);
    }
    int clientId = atoi(argv[1]);
    REDEV_ALWAYS_ASSERT(clientId >= -1 && clientId <= 5);
    const auto meshFile = argv[2];
    const auto classPartitionFile = argv[3];

    if (clientId >= 4) {
      test_rcb_routing(lib, /*is_server=*/clientId == 4);
      return 0;
    }
    Omega_h::Mesh mesh = Omega_h::binary::read(meshFile, world);
    adios2::Params params{{"Streaming", "On"},
                          {"OpenTimeoutSecs", "60"},
                          {"BeginStepPollingFrequencySecs", "0.05"}};
    MPI_Comm mpi_comm = lib.world()->get_impl();

    switch (clientId) {
      case -1:
        server(mpi_comm, mesh, "lin_field_comm", 1, params, classPartitionFile);
        break;
      case 0: client1(mpi_comm, mesh, "lin_field_comm", 1, params); break;
      case 1: client2(mpi_comm, mesh, "lin_field_comm", 1, params); break;
      case 2:
        test_shared_layout(lib, meshFile, classPartitionFile,
                           /*is_server=*/true);
        break;
      case 3:
        test_shared_layout(lib, meshFile, classPartitionFile,
                           /*is_server=*/false);
        break;
      default:
        std::cerr << "Unhandled client id (should be -1,0,1,2,3)\n";
        exit(EXIT_FAILURE);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Exception caught in main: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught in main" << std::endl;
    return 1;
  }
}
