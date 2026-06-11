#include <petscksp.h>

#include "pcms/transfer/conservative_projection_solver.hpp"
#include "pcms/transfer/calculate_load_vector.hpp"
#include "pcms/transfer/calculate_mass_matrix.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/transfer/petsc_utils.hpp"
#include <Kokkos_Core.hpp>

namespace pcms
{

static Omega_h::Reals vecToOmegaHReals(Vec vec)
{
  PetscInt n = 0;
  PetscErrorCode ierr = VecGetSize(vec, &n);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  const PetscScalar* array = nullptr;
  ierr = VecGetArrayRead(vec, &array);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  auto values_host = Omega_h::HostWrite<Omega_h::Real>(n);
  for (PetscInt i = 0; i < n; ++i) {
    values_host[i] = array[i];
  }

  ierr = VecRestoreArrayRead(vec, &array);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  return Omega_h::Reals(values_host);
}

// One-shot solve used by the legacy path and solveGalerkinProjection.
// Creates, solves, and destroys the KSP in a single call.
static Vec solveLinearSystem(Mat A, Vec b)
{
  PetscInt m, n;
  PetscErrorCode ierr = MatGetSize(A, &m, &n);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  Vec x;
  ierr = createSeqVec(PETSC_COMM_WORLD, n, &x);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPSetOperators(ksp, A, A);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPSetFromOptions(ksp);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPSetUp(ksp);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPSolve(ksp, b, x);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPDestroy(&ksp);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  return x;
}

// ---------------------------------------------------------------------------
// GalerkinProjectionSolver
// ---------------------------------------------------------------------------

GalerkinProjectionSolver::GalerkinProjectionSolver(
  const BilinearFormIntegrator& mass_integrator)
{
  Mat A = mass_integrator.GetMatrix();

  PetscInt m = 0, n = 0;
  PetscErrorCode ierr = MatGetSize(A, &m, &n);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  nverts_ = m;

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp_);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPSetOperators(ksp_, A, A);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPSetFromOptions(ksp_);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPSetUp(ksp_);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

GalerkinProjectionSolver::~GalerkinProjectionSolver()
{
  if (ksp_) {
    KSPDestroy(&ksp_);
  }
  // mat_ is owned by the BilinearFormIntegrator; KSP holds its own reference.
}

Omega_h::Reals GalerkinProjectionSolver::Solve(
  LinearFormIntegrator& rhs_integrator, const PointEvaluator<Real>& evaluator,
  const Field<Real>& source_field) const
{
  const std::size_t num_pts = rhs_integrator.GetIntegrationPoints()
                                .GetCoordinates()
                                .GetValues()
                                .extent(0);
  Kokkos::View<Real**, DeviceMemorySpace> sampled_values("rhs_sampled", num_pts,
                                                         1);
  evaluator.Evaluate(source_field, MakeRank2View(sampled_values));

  return Solve(rhs_integrator, MakeConstRank2View(sampled_values));
}

Omega_h::Reals GalerkinProjectionSolver::Solve(
  LinearFormIntegrator& rhs_integrator,
  Rank2View<const Real, DeviceMemorySpace> sampled_values) const
{
  rhs_integrator.Assemble(sampled_values);
  Vec rhs_vector = rhs_integrator.GetVector();

  Vec solution = nullptr;
  PetscErrorCode ierr = createSeqVec(PETSC_COMM_WORLD, nverts_, &solution);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  ierr = KSPSolve(ksp_, rhs_vector, solution);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  auto result = vecToOmegaHReals(solution);

  ierr = VecDestroy(&solution);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  return result;
}

// ---------------------------------------------------------------------------
// Legacy one-shot paths
// ---------------------------------------------------------------------------

Omega_h::Reals solveGalerkinProjection(Omega_h::Mesh& target_mesh,
                                       Omega_h::Mesh& source_mesh,
                                       const IntersectionResults& intersection,
                                       const Omega_h::Reals& source_values)
{
  if ((PetscInt)source_values.size() !=
      source_mesh.coords().size() / source_mesh.dim()) {
    throw std::runtime_error("source_values length mismatch");
  }

  Mat mass;
  PetscErrorCode ierr = calculateMassMatrix(target_mesh, &mass);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  Vec vec;
  ierr = calculateLoadVector(target_mesh, source_mesh, intersection,
                             source_values, &vec);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  Vec x = solveLinearSystem(mass, vec);
  auto solution_vector = vecToOmegaHReals(x);

  ierr = VecDestroy(&x);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = MatDestroy(&mass);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDestroy(&vec);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  return solution_vector;
}

Omega_h::Reals rhsVectorMI(Omega_h::Mesh& target_mesh,
                           Omega_h::Mesh& source_mesh,
                           const IntersectionResults& intersection,
                           const Omega_h::Reals& source_values)
{
  Vec vec;
  PetscErrorCode ierr = calculateLoadVector(target_mesh, source_mesh,
                                            intersection, source_values, &vec);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  auto rhsvector = vecToOmegaHReals(vec);

  ierr = VecDestroy(&vec);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  return rhsvector;
}

} // namespace pcms
