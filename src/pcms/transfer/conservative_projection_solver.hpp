#ifndef PCMS_TRANSFER_CONSERVATIVE_PROJECTION_SOLVER_HPP
#define PCMS_TRANSFER_CONSERVATIVE_PROJECTION_SOLVER_HPP

#include <Kokkos_Core.hpp>
#include <Omega_h_array.hpp>
#include <petscksp.h>

#include <pcms/transfer/bilinear_form_integrator.hpp>
#include <pcms/utility/arrays.hpp>

namespace pcms::transfer
{

// Cached Galerkin projection solver for M*x = b.
//
// The mass matrix is obtained from mass_integrator (which owns and assembled
// it) and the KSP is set up (preconditioned/factored) at construction. Each
// Solve() call only copies the right-hand side into a PETSc vector and calls
// KSPSolve, reusing the cached factorization. The right-hand side is
// b = E^T W f, accumulated by PointEvaluator::EvaluateTranspose.
//
// Non-copyable and non-movable because it owns PETSc handles.
class GalerkinProjectionSolver
{
public:
  // Retrieves the assembled matrix from mass_integrator and sets up the KSP.
  // PETSc reference-counts the matrix, so mass_integrator need not outlive
  // this solver.
  explicit GalerkinProjectionSolver(
    const BilinearFormIntegrator& mass_integrator);
  ~GalerkinProjectionSolver();

  GalerkinProjectionSolver(const GalerkinProjectionSolver&) = delete;
  GalerkinProjectionSolver& operator=(const GalerkinProjectionSolver&) = delete;

  // Solves M*x = rhs with the cached KSP. `rhs` has one entry per row of M
  // (the layout's DOF holders with ghosts, in its global-to-local order).
  Omega_h::Reals Solve(Kokkos::View<const Real*, DeviceMemorySpace> rhs) const;

private:
  PetscInt nverts_;
  KSP ksp_ = nullptr;
};

} // namespace pcms::transfer

#endif // PCMS_TRANSFER_CONSERVATIVE_PROJECTION_SOLVER_HPP
