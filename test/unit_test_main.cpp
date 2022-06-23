#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <wdmcpl/point_search.h>
#include <Omega_h_build.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <iostream>
#include <Omega_h_library.hpp>

// FIXME spelling
//Omega_h::Library omega_h_library{Omega_h::DoNotIntialize{}};
//Omega_h::Library omega_h_library{nullptr, nullptr};
Omega_h::Library* omega_h_library = nullptr;

int main( int argc, char* argv[] )
{
  //MPI_Init(&argc, &argv);
  std::cout<<"Initialize"<<std::endl;
  Omega_h::Library lib(&argc,&argv);
  omega_h_library = &lib;
  //[[maybe_unused]] auto world = omega_h_library.world();
  Kokkos::ScopeGuard kokkos{};
  // initialize kokkos/redev/mpi/omega_h/whatever
  int result = Catch::Session().run(argc, argv);
  //MPI_Finalize();
  return result;
}
