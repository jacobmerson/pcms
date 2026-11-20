#include <catch2/catch_session.hpp>
#include <pcms/configuration.hpp>
#ifdef PCMS_ENABLE_MPI
#include <mpi.h>
#endif
#include <Kokkos_Core.hpp>

#ifdef PCMS_ENABLE_CONSERVATIVE_PROJECTION
#include <petscsys.h>
#endif

int main(int argc, char* argv[])
{
#ifdef PCMS_ENABLE_MPI
  MPI_Init(&argc, &argv);
#endif
  int result = 0;
  {
    // PETSc uses Kokkos, so initialize Kokkos before PETSc when enabled.
    Kokkos::ScopeGuard kokkos{argc, argv};
#ifdef PCMS_ENABLE_CONSERVATIVE_PROJECTION
    PetscBool petsc_initialized = PETSC_FALSE;
    PetscBool petsc_initialized_by_main = PETSC_FALSE;
    PetscInitialized(&petsc_initialized);
    if (!petsc_initialized) {
      PetscInitialize(&argc, &argv, nullptr, nullptr);
      petsc_initialized_by_main = PETSC_TRUE;
    }
    // petsc must be finalized in the kokkos ScopeGuard scope kokkos needs to
    // deallocate after petsc is done finalizing.
    result = Catch::Session().run(argc, argv);
    if (petsc_initialized_by_main) {
      PetscFinalize();
    }
#else
    result = Catch::Session().run(argc, argv);
#endif
  }
#ifdef PCMS_ENABLE_MPI
  MPI_Finalize();
#endif
  return result;
}
