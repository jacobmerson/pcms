#ifndef PCMS_TRANSFER_BILINEAR_FORM_INTEGRATOR_H
#define PCMS_TRANSFER_BILINEAR_FORM_INTEGRATOR_H

#include <petscmat.h>

namespace pcms
{

class BilinearFormIntegrator
{
public:
  // Returns the internally-assembled matrix. The integrator owns the matrix
  // and its lifetime must exceed any KSP that references it (PETSc's reference
  // counting keeps the matrix alive until the KSP is destroyed).
  virtual Mat GetMatrix() const noexcept = 0;

  virtual ~BilinearFormIntegrator() noexcept = default;
};

} // namespace pcms

#endif // PCMS_TRANSFER_BILINEAR_FORM_INTEGRATOR_H
