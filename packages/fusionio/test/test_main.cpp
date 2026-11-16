#include <catch2/catch_session.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos{argc, argv};
    result = Catch::Session().run(argc, argv);
  }
  MPI_Finalize();
  return result;
}
