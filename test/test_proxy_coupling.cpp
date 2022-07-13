#include <Omega_h_mesh.hpp>
#include <iostream>
#include <wdmcpl.h>
#include <wdmcpl/types.h>
#include <Omega_h_file.hpp>

// Serializer is used in a two pass algorithm. Must check that the buffer size
// >0 and return the number of entries.
struct SerializeOmegaH
{
  int operator()(std::string_view , wdmcpl::Field*, nonstd::span<wdmcpl::Real>) {}
  int operator()(std::string_view , wdmcpl::Field*, nonstd::span<wdmcpl::LO>) {}
  int operator()(std::string_view , wdmcpl::Field*, nonstd::span<wdmcpl::GO>) {}
};
struct DeserializeOmegaH
{
  void operator()(std::string_view, wdmcpl::Field*, nonstd::span<const wdmcpl::Real>) {}
  void operator()(std::string_view, wdmcpl::Field*, nonstd::span<const wdmcpl::LO>) {}
  void operator()(std::string_view, wdmcpl::Field*, nonstd::span<const wdmcpl::GO>) {}
};

void xgc_delta_f(MPI_Comm comm)
{
  wdmcpl::Coupler cpl("proxy_couple", wdmcpl::ProcessType::Client);
  wdmcpl::Field density;
  cpl.add_mesh_partition("xgc", comm, redev::ClassPtn{});
  cpl.add_field<wdmcpl::Real>("xgc", "density", &density);
  cpl.ScatterField("density", SerializeOmegaH{});
  // get updated field data from coupling server
  cpl.GatherField("density", DeserializeOmegaH{});
}
void xgc_total_f() {}
void coupler()
{
  // while not-done
  //    receive n=0 data from total_f
  //    send n=0 data to delta_f
  //    receive potential in overlap from total-f/delta-f
  //    combine potential in overlap region
  //    send potential in overlap to total-f/delta-f
  //    receive density in overlap from total-f/delta-f
  //    combine density in overlap region
  //    send density in overlap to total-f/delta-f
}

int main(int argc, char** argv)
{
  /*
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();
  const int rank = world->rank();
  if(argc != 4) {
    if(!rank) {
      std::cerr << "Usage: " << argv[0] << " <clientId=-1|0|1>
 /path/to/omega_h/mesh /path/to/partitionFile.cpn\n";
    }
    exit(EXIT_FAILURE);
  }
  OMEGA_H_CHECK(argc == 4);
  const auto clientId = atoi(argv[1]);
  REDEV_ALWAYS_ASSERT(clientId >= -1 && clientId <=1);
  const auto meshFile = argv[2];
  const auto classPartitionFile = argv[3];
  Omega_h::Mesh mesh(&lib);
  Omega_h::binary::read(meshFile, lib.world(), &mesh);
  const std::string name = "meshVtxIds";
  switch (clientId) {
    case -1:
      coupler(mesh,classPartitionFile);
      coupler();
      break;
    case 0:
      xgc_delta_f(mesh);
      break;
    case 1:
      xgc_total_f(mesh);
      break;
    default:
      std::cerr<<"Unhandled client id (should be -1, 0,1)\n";
      exit(EXIT_FAILURE);
  }
 // if(clientId == -1) { //rendezvous
 //   coupler(mesh,name,classPartitionFile);
 // } else {
 //   client(mesh,name,clientId);
 // }
   */
  return 0;
}