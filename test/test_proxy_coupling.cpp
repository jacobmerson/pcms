#include <Omega_h_mesh.hpp>
#include <iostream>
#include <wdmcpl.h>
#include <wdmcpl/types.h>
#include <Omega_h_file.hpp>
#include <redev_variant_tools.h>
#include "test_support.h"

// Serializer is used in a two pass algorithm. Must check that the buffer size
// >0 and return the number of entries.
struct SerializeOmegaH
{
  SerializeOmegaH(Omega_h::Mesh& mesh) : mesh(mesh) {}
  int operator()(std::string_view, wdmcpl::Field*, nonstd::span<wdmcpl::Real>)
  {
  }
  int operator()(std::string_view, wdmcpl::Field*, nonstd::span<wdmcpl::LO>) {}
  int operator()(std::string_view, wdmcpl::Field*, nonstd::span<wdmcpl::GO>) {}
  Omega_h::Mesh mesh;
};
struct DeserializeOmegaH
{
  DeserializeOmegaH(Omega_h::Mesh& mesh) : mesh(mesh) {}
  void operator()(std::string_view, wdmcpl::Field*,
                  nonstd::span<const wdmcpl::Real>)
  {
  }
  void operator()(std::string_view, wdmcpl::Field*,
                  nonstd::span<const wdmcpl::LO>)
  {
  }
  void operator()(std::string_view, wdmcpl::Field*,
                  nonstd::span<const wdmcpl::GO>)
  {
  }
  Omega_h::Mesh& mesh;
};

struct OmegaHPartitionRankCount
{
  OmegaHPartitionRankCount(Omega_h::Mesh& mesh) : mesh(mesh) {}
  std::map<int, int> operator()(std::string_view, wdmcpl::Field*,
                                const redev::Partition& partition)
  {
    auto ohComm = mesh.comm();
    const auto rank = ohComm->rank();
    // transfer vtx classification to host
    auto classIds = mesh.get_array<Omega_h::ClassId>(0, "class_id");
    auto classIds_h = Omega_h::HostRead(classIds);
    auto classDims = mesh.get_array<Omega_h::I8>(0, "class_dim");
    auto classDims_h = Omega_h::HostRead(classDims);
    auto isOverlap =
      mesh.has_tag(0, "isOverlap")
        ? mesh.get_array<Omega_h::I8>(0, "isOverlap")
        : Omega_h::Read<Omega_h::I8>(
            classIds.size(), 1, "isOverlap"); // no mask for overlap vertices
    auto isOverlap_h = Omega_h::HostRead(isOverlap);
    // count number of vertices going to each destination process by calling
    // getRank - degree array
    std::map<int, int> destRankCounts;
    auto ranks = std::visit([](auto&& p) { return p.GetRanks(); }, partition);
    for (auto rank : ranks) {
      destRankCounts[rank] = 0;
    }
    for (auto i = 0; i < classIds_h.size(); i++) {
      if (isOverlap_h[i]) {
        auto dr = std::visit(
          wdmcpl::overloaded{
            [&classDims_h, &classIds_h, &i](const redev::ClassPtn& ptn) {
              const auto ent =
                redev::ClassPtn::ModelEnt({classDims_h[i], classIds_h[i]});
              return ptn.GetRank(ent);
            },
            [](const redev::RCBPtn& ptn) {
              std::cerr << "RCB partition not handled yet\n";
              std::exit(EXIT_FAILURE);
              return 0;
            }},
          partition);
        assert(destRankCounts.count(dr));
        destRankCounts[dr]++;
      }
    }
    return destRankCounts;
  }
  Omega_h::Mesh& mesh;
};

redev::ClassPtn setupServerPartition(Omega_h::Mesh& mesh, std::string_view cpnFileName) {
  namespace ts = test_support;
  auto ohComm = mesh.comm();
  const auto facePartition = !ohComm->rank() ? ts::readClassPartitionFile(cpnFileName) :
                             ts::ClassificationPartition();
  ts::migrateMeshElms(mesh, facePartition);
  auto ptn = ts::CreateClassificationPartition(mesh);
  return redev::ClassPtn(MPI_COMM_WORLD, ptn.ranks, ptn.modelEnts);
}

void xgc_delta_f(MPI_Comm comm, Omega_h::Mesh& mesh)
{
  wdmcpl::Coupler cpl("proxy_couple", wdmcpl::ProcessType::Client);
  cpl.add_mesh_partition("xgc", comm, redev::ClassPtn{});
  cpl.add_field<wdmcpl::Real>("xgc", "delta_f_density", nullptr,
                              OmegaHPartitionRankCount{mesh});
  cpl.ScatterField("delta_f_density", SerializeOmegaH{mesh});
  // get updated field data from coupling server
  cpl.GatherField("delta_f_density", DeserializeOmegaH{mesh});
}
void xgc_total_f(MPI_Comm comm, Omega_h::Mesh& mesh) {
  wdmcpl::Coupler cpl("proxy_couple", wdmcpl::ProcessType::Client);
  cpl.add_mesh_partition("xgc", comm, redev::ClassPtn{});
  cpl.add_field<wdmcpl::Real>("xgc", "total_f_density", nullptr,
                              OmegaHPartitionRankCount{mesh});
  cpl.ScatterField("total_f_density", SerializeOmegaH{mesh});
  // get updated field data from coupling server
  cpl.GatherField("total_f_density", DeserializeOmegaH{mesh});
}
void coupler(MPI_Comm comm, Omega_h::Mesh& mesh, std::string_view cpn_file)
{

  wdmcpl::Coupler cpl("proxy_couple", wdmcpl::ProcessType::Server);
  cpl.add_mesh_partition("xgc", comm, setupServerPartition(mesh,cpn_file) );
  cpl.add_field<wdmcpl::Real>("xgc", "density", nullptr,
                              OmegaHPartitionRankCount{mesh});
  // currently, gather field is just receiving. But it's actually
  // supposed to receive all of the application constituent fields
  // and gather them onto the internal representation
  cpl.GatherField("delta_f_density", DeserializeOmegaH{mesh});
  cpl.GatherField("total_f_density", DeserializeOmegaH{mesh});
  // combine fields
  // currently missing link between application fields and combined state
  // This is supposed to send out the internal data to the application.
  // This includes a field transfer back to the native meshes
  cpl.ScatterField("density", SerializeOmegaH{mesh});
}

int main(int argc, char** argv)
{
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  const int rank = world->rank();
  if (argc != 4) {
    if (!rank) {
      std::cerr << "Usage: " << argv[0]
                << " <clientId=-1|0|1> /path/to/omega_h/mesh "
                   "/path/to/partitionFile.cpn\n";
    }
    exit(EXIT_FAILURE);
  }
  OMEGA_H_CHECK(argc == 4);
  const auto clientId = atoi(argv[1]);
  REDEV_ALWAYS_ASSERT(clientId >= -1 && clientId <= 1);
  const auto meshFile = argv[2];
  const auto classPartitionFile = argv[3];
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(meshFile, lib.world(), &mesh);
  MPI_Comm mpi_comm = lib.world()->get_impl();
  const std::string name = "meshVtxIds";
  switch (clientId) {
    case -1:
      coupler(mpi_comm, mesh, classPartitionFile);
      break;
    case 0: xgc_delta_f(mpi_comm, mesh); break;
    case 1: xgc_total_f(mpi_comm, mesh); break;
    default:
      std::cerr << "Unhandled client id (should be -1, 0,1)\n";
      exit(EXIT_FAILURE);
  }
  return 0;
}