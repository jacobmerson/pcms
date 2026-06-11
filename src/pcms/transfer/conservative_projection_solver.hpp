#ifndef PCMS_TRANSFER_CONSERVATIVE_PROJECTION_SOLVER_HPP
#define PCMS_TRANSFER_CONSERVATIVE_PROJECTION_SOLVER_HPP

#include <Omega_h_array.hpp>
#include <petscksp.h>

#include <pcms/field/field.h>
#include <pcms/field/point_evaluator.h>
#include <pcms/transfer/bilinear_form_integrator.hpp>
#include <pcms/transfer/linear_form_integrator.hpp>
#include <pcms/transfer/mesh_intersection.hpp>

namespace pcms
{

// Cached Galerkin projection solver for M*x = f.
//
// The mass matrix is obtained from the integrator (which owns and assembled it)
// and the KSP is set up (preconditioned/factored) at construction. Each Solve()
// call only assembles a fresh RHS vector and calls KSPSolve, reusing the cached
// factorization.
//
// The solver is decoupled from any particular integration strategy: it works
// with any LinearFormIntegrator passed to Solve(), not just the mesh-
// intersection-based one. This is the intended use of IntegrationPointSet —
// the solver knows nothing about how those points were generated.
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

  // Evaluates source_field at rhs_integrator's integration points via
  // evaluator, assembles the RHS, solves M*x = f using the cached KSP, and
  // returns nodal values on the target mesh.
  //
  // May be called repeatedly with different source fields or even different
  // rhs_integrators, provided the target mesh (and therefore M) is unchanged.
  Omega_h::Reals Solve(LinearFormIntegrator& rhs_integrator,
                       const PointEvaluator<Real>& evaluator,
                       const Field<Real>& source_field) const;

  // Assembles the RHS from values already sampled at rhs_integrator's
  // integration points (shape [num_points][num_components]) and solves
  // M*x = f using the cached KSP. Used by workflows that pre-process the
  // sampled values (e.g. control-variate residuals) before assembly.
  Omega_h::Reals Solve(
    LinearFormIntegrator& rhs_integrator,
    Rank2View<const Real, DeviceMemorySpace> sampled_values) const;

private:
  PetscInt nverts_;
  KSP ksp_ = nullptr;
};

// One-shot Galerkin projection (reference / legacy path).
// Builds and tears down the matrix and KSP on every call.
// Prefer GalerkinProjectionSolver for repeated use on the same mesh pair.
Omega_h::Reals solveGalerkinProjection(Omega_h::Mesh& target_mesh,
                                       Omega_h::Mesh& source_mesh,
                                       const IntersectionResults& intersection,
                                       const Omega_h::Reals& source_values);

Omega_h::Reals rhsVectorMI(Omega_h::Mesh& target_mesh,
                           Omega_h::Mesh& source_mesh,
                           const IntersectionResults& intersection,
                           const Omega_h::Reals& source_values);

} // namespace pcms

#endif // PCMS_TRANSFER_CONSERVATIVE_PROJECTION_SOLVER_HPP
