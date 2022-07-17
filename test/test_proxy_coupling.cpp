#include <Omega_h_mesh.hpp>
#include <iostream>
#include <wdmcpl.h>
#include <wdmcpl/types.h>
#include <Omega_h_file.hpp>
#include <OMega_h_for.hpp>
#include <redev_variant_tools.h>
#include "test_support.h"


/**
 * return 1 if the specificed model entity is part of the overlap region, 0
 * otherwise
 */
OMEGA_H_DEVICE Omega_h::I8 isModelEntInOverlap(const int dim, const int id) {
  //the TOMMS generated geometric model has
  //entity IDs that increase with the distance
  //from the magnetic axis
  if (dim == 2 && (id >= 22 && id <= 34) ) {
    return 1;
  } else if (dim == 1 && (id >= 21 && id <= 34) ) {
    return 1;
  } else if (dim == 0 && (id >= 21 && id <= 34) ) {
    return 1;
  }
  return 0;
}

/**
 * Create the tag 'isOverlap' for each mesh vertex whose value is 1 if the
 * vertex is classified on a model entity in the closure of the geometric model
 * faces forming the overlap region; the value is 0 otherwise. OnlyIncludesOverlapping and owned verts
 */
Omega_h::Read<Omega_h::I8> markOverlapMeshEntities(Omega_h::Mesh& mesh) {
  //transfer vtx classification to host
  auto classIds = mesh.get_array<Omega_h::ClassId>(0, "class_id");
  auto classDims = mesh.get_array<Omega_h::I8>(0, "class_dim");
  auto isOverlap = Omega_h::Write<Omega_h::I8>(classIds.size(), "isOverlap");
  auto markOverlap = OMEGA_H_LAMBDA(int i) {
    isOverlap[i] = isModelEntInOverlap(classDims[i], classIds[i]);
  };
  Omega_h::parallel_for(classIds.size(), markOverlap);
  auto isOwned = mesh.owned(0);
  // try masking out to only owned entities
  Omega_h::parallel_for(isOverlap.size(),OMEGA_H_LAMBDA(int i){
    isOverlap[i] = (isOwned[i] && isOverlap[i]);
  });

  auto isOverlap_r = Omega_h::read(isOverlap);
  mesh.add_tag(0, "isOverlap", 1, isOverlap_r);
  return isOverlap_r;
}
Omega_h::HostRead<Omega_h::I8> markMeshOverlapRegion(Omega_h::Mesh& mesh) {
  auto isOverlap = markOverlapMeshEntities(mesh);
  return Omega_h::HostRead(isOverlap);
}

struct SerializeOmegaHGids
{
  SerializeOmegaHGids(Omega_h::Mesh& mesh, Omega_h::HostRead<Omega_h::I8> is_overlap_h) : mesh_(mesh), is_overlap_h_(is_overlap_h) {}
  template<typename T>
  int operator()(std::string_view, wdmcpl::Field*, nonstd::span<T> buffer ,
                 nonstd::span<const wdmcpl::LO> permutation)
  {
    if(buffer.size() == 0) {
      return is_overlap_h_.size();
    }
    //WDMCPL_ALWAYS_ASSERT(buffer.size() == is_overlap_h_.size());
    auto gids = mesh_.globals(0);
    auto gids_h = Omega_h::HostRead(gids);
    int j=0;
    for(size_t i=0; i<gids_h.size(); i++) {
      if( is_overlap_h_[i] ) {
        buffer[permutation[j++]] = gids_h[i];
      }
    }
    return is_overlap_h_.size();
  }
  Omega_h::Mesh mesh_;
  Omega_h::HostRead<Omega_h::I8> is_overlap_h_;
};


// Serializer is used in a two pass algorithm. Must check that the buffer size
// >0 and return the number of entries.
struct SerializeOmegaH
{
  SerializeOmegaH(Omega_h::Mesh& mesh, Omega_h::HostRead<Omega_h::I8> is_overlap_h) : mesh_(mesh), is_overlap_h_(is_overlap_h) {}
  template<typename T>
  int operator()(std::string_view name, wdmcpl::Field*, nonstd::span<T> buffer ,
                 nonstd::span<const wdmcpl::LO> permutation)
  {
    if(buffer.size() == 0) {
      return is_overlap_h_.size();
    }
    //WDMCPL_ALWAYS_ASSERT(buffer.size() == is_overlap_h_.size());
    const auto array = mesh_.get_array<T>(0, std::string(name));
    const auto array_h = Omega_h::HostRead(array);
    int j=0;
    for(size_t i=0; i<array_h.size(); i++) {
      if( is_overlap_h_[i] ) {
        buffer[permutation[j++]] = array_h[i];
      }
    }
    return is_overlap_h_.size();
  }
  Omega_h::Mesh mesh_;
  Omega_h::HostRead<Omega_h::I8> is_overlap_h_;
};
struct DeserializeOmegaH
{
  DeserializeOmegaH(Omega_h::Mesh& mesh) : mesh(mesh) {}
  template <typename T>
  void operator()(std::string_view, wdmcpl::Field*,
                  nonstd::span<const T>, nonstd::span<const wdmcpl::LO> )
  {
  }
  Omega_h::Mesh& mesh;
};

struct OmegaHReversePartition
{
  OmegaHReversePartition(Omega_h::Mesh& mesh) : mesh(mesh) {}
  wdmcpl::ReversePartitionMap operator()(std::string_view, wdmcpl::Field*,
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
    wdmcpl::ReversePartitionMap reverse_partition;
    wdmcpl::LO count = 0;
    for (auto i = 0; i < classIds_h.size(); i++) {
      if (isOverlap_h[i]) {
        auto dr = std::visit(
          redev::overloaded{
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
        reverse_partition[dr].push_back(count++);
      }
    }
    return reverse_partition;
  }
  Omega_h::Mesh& mesh;
};

redev::ClassPtn setupServerPartition(Omega_h::Mesh& mesh,
                                     std::string_view cpnFileName)
{
  namespace ts = test_support;
  auto ohComm = mesh.comm();
  const auto facePartition = !ohComm->rank()
                               ? ts::readClassPartitionFile(cpnFileName)
                               : ts::ClassificationPartition();
  ts::migrateMeshElms(mesh, facePartition);
  auto ptn = ts::CreateClassificationPartition(mesh);
  return redev::ClassPtn(MPI_COMM_WORLD, ptn.ranks, ptn.modelEnts);
}

void xgc_delta_f(MPI_Comm comm, Omega_h::Mesh& mesh)
{
  std::cerr<<"Start\n";
  wdmcpl::Coupler cpl("proxy_couple", wdmcpl::ProcessType::Client);
  /*
  cpl.AddMeshPartition("xgc", comm, redev::ClassPtn{});
  cpl.AddField<wdmcpl::Real>("xgc", "delta_f_density", nullptr,
                             OmegaHReversePartition{mesh});
  cpl.SendField("delta_f_density", SerializeOmegaH{mesh});
  // get updated field data from coupling server
  cpl.ReceiveField("delta_f_density", DeserializeOmegaH{mesh});
  */
  std::cerr<<"1\n";
  cpl.AddMeshPartition("xgc", comm, redev::ClassPtn{});
  //std::cerr<<"2\n";
  cpl.AddField<wdmcpl::GO>("xgc", "gids", nullptr,
                             OmegaHReversePartition{mesh});
  std::cerr<<"3\n";
  //auto is_overlap_h = markMeshOverlapRegion(mesh);
  std::cerr<<"4\n";
  //cpl.SendField("gids", SerializeOmegaHGids{mesh,is_overlap_h});
  // get updated field data from coupling server
  //cpl.ReceiveField("gids", DeserializeOmegaH{mesh});
}
void xgc_total_f(MPI_Comm comm, Omega_h::Mesh& mesh)
{
  /*
  wdmcpl::Coupler cpl("proxy_couple", wdmcpl::ProcessType::Client);
  cpl.AddMeshPartition("xgc", comm, redev::ClassPtn{});
  cpl.AddField<wdmcpl::Real>("xgc", "total_f_density", nullptr,
                             OmegaHReversePartition{mesh});
  cpl.SendField("total_f_density", SerializeOmegaH{mesh});
  // get updated field data from coupling server
  cpl.ReceiveField("total_f_density", DeserializeOmegaH{mesh});
*/
}
void coupler(MPI_Comm comm, Omega_h::Mesh& mesh, std::string_view cpn_file)
{

  wdmcpl::Coupler cpl("proxy_couple", wdmcpl::ProcessType::Server);

  cpl.AddMeshPartition("xgc", comm, setupServerPartition(mesh, cpn_file));

  cpl.AddField<wdmcpl::GO>("xgc", "gids", nullptr,
                             OmegaHReversePartition{mesh});
  auto is_overlap_h = markMeshOverlapRegion(mesh);
  cpl.ReceiveField("xgc", DeserializeOmegaH{mesh});
  //cpl.SendField("xgc", SerializeOmegaHGids{mesh,is_overlap_h});

  /*
  cpl.AddMeshPartition("xgc", comm, setupServerPartition(mesh, cpn_file));
  cpl.AddField<wdmcpl::Real>("xgc", "density", nullptr,
                             OmegaHReversePartition{mesh});
  // Gather Field
  {
    cpl.ReceiveField("delta_f_density", DeserializeOmegaH{mesh});
    cpl.ReceiveField("total_f_density", DeserializeOmegaH{mesh});
    // TODO: field_transfer
    // TODO: combine fields
  }

  // Scatter Field
  {
    // TODO: field_transfer to native
    cpl.SendField("delta_f_density", SerializeOmegaH{mesh});
    cpl.SendField("total_f_density", SerializeOmegaH{mesh});
  }
   */
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
    case -1: coupler(mpi_comm, mesh, classPartitionFile); break;
    case 0: xgc_delta_f(mpi_comm, mesh); break;
    case 1: xgc_total_f(mpi_comm, mesh); break;
    default:
      std::cerr << "Unhandled client id (should be -1, 0,1)\n";
      exit(EXIT_FAILURE);
  }
  return 0;
}