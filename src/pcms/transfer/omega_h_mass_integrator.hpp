#ifndef PCMS_TRANSFER_OMEGA_H_MASS_INTEGRATOR_HPP
#define PCMS_TRANSFER_OMEGA_H_MASS_INTEGRATOR_HPP

#include "pcms/field/function_space.h"
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/transfer/bilinear_form_integrator.hpp"
#include <memory>
#include <vector>

namespace pcms
{

// Mass-matrix bilinear form integrator for Lagrange spaces on Omega_h 2D
// simplex meshes. Supports order-0 (diagonal, M_ee = area(e)) and order-1
// (consistent 3x3 element blocks) target spaces.
//
// The matrix is assembled once at construction into a PETSc sparse AIJ matrix.
// Sparsity is derived directly from the element node GIDs — no separate COO
// utility is needed. The integrator owns the matrix; callers receive a
// non-owning handle via GetMatrix(). PETSc reference-counts the matrix, so a
// KSP that calls KSPSetOperators(GetMatrix()) safely outlives this object.
class OmegaHMassIntegrator : public BilinearFormIntegrator
{
public:
  explicit OmegaHMassIntegrator(const FunctionSpace& target_space);
  OmegaHMassIntegrator(
    std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
    CoordinateSystem coordinate_system);
  ~OmegaHMassIntegrator();

  Mat GetMatrix() const noexcept override;

private:
  Mat mat_ = nullptr;
};

// Builds an OmegaHMassIntegrator for the given target space.
// target_space must use OmegaHLagrangeLayout, scalar, Cartesian, order-0 or
// order-1.
std::unique_ptr<BilinearFormIntegrator> BuildOmegaHMassIntegrator(
  const FunctionSpace& target_space);

} // namespace pcms

#endif // PCMS_TRANSFER_OMEGA_H_MASS_INTEGRATOR_HPP
