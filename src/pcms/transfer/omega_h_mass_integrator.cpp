#include "pcms/transfer/omega_h_mass_integrator.hpp"
#include "pcms/transfer/mass_matrix_integrator.hpp"
#include "pcms/transfer/omega_h_form_integrator_utils.hpp"
#include "pcms/transfer/petsc_utils.hpp"
#include "pcms/utility/assert.h"
#include "pcms/utility/memory_spaces.h"
#include <KokkosController.hpp>
#include <MeshField.hpp>
#include <MeshField_Element.hpp>
#include <Omega_h_for.hpp>
#include <petscksp.h>

namespace pcms
{

OmegaHMassIntegrator::OmegaHMassIntegrator(const FunctionSpace& target_space)
  : OmegaHMassIntegrator(
      std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
        target_space.GetLayout()),
      target_space.GetCoordinateSystem())
{
}

OmegaHMassIntegrator::OmegaHMassIntegrator(
  std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
  CoordinateSystem coordinate_system)
{
  detail::CheckOmegaHScalarP1Layout(coordinate_system, target_layout,
                                    "OmegaHMassIntegrator", "target");

  Omega_h::Mesh& mesh = target_layout->GetMesh();
  PCMS_ALWAYS_ASSERT(mesh.dim() == 2);
  PCMS_ALWAYS_ASSERT(mesh.family() == OMEGA_H_SIMPLEX);

  // Build per-element mass matrices using MeshField.
  MeshField::OmegahMeshField<DefaultExecutionSpace, 2,
                              MeshField::KokkosController>
    omf(mesh);
  auto coordField = omf.getCoordField();
  const auto [shp, map] = MeshField::Omegah::getTriangleElement<1>(mesh);
  MeshField::FieldElement coordFe(mesh.nelems(), coordField, shp, map);
  auto elm_mass_dev = buildElementMassMatrix(mesh, coordFe);

  // Build COO sparsity pattern on device: each element contributes a 3x3 block.
  const int nelems = mesh.nelems();
  const auto& faces2nodes = mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;
  const auto device_gids = target_layout->GetGids();
  const GO* gids_ptr = device_gids.data_handle();

  const PetscInt nnz = static_cast<PetscInt>(nelems) * 9;
  Kokkos::View<PetscInt*, DeviceMemorySpace> coo_rows("mass_coo_rows", nnz);
  Kokkos::View<PetscInt*, DeviceMemorySpace> coo_cols("mass_coo_cols", nnz);
  Kokkos::parallel_for(
    "mass_coo_pattern", nelems, KOKKOS_LAMBDA(int e) {
      const auto verts = Omega_h::gather_verts<3>(faces2nodes, e);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          const int idx = e * 9 + i * 3 + j;
          coo_rows(idx) = static_cast<PetscInt>(gids_ptr[verts[i]]);
          coo_cols(idx) = static_cast<PetscInt>(gids_ptr[verts[j]]);
        }
      }
    });

  // Create sparse matrix, preallocate with COO pattern, then bulk-set values.
  // elm_mass_dev is in the same element-major order as coo_rows/coo_cols, so
  // it can be passed directly to MatSetValuesCOO — no host copy needed.
  const PetscInt nverts = static_cast<PetscInt>(mesh.nverts());
  PetscErrorCode ierr =
    createSeqAIJMat(PETSC_COMM_WORLD, nverts, nverts, 0, nullptr, &mat_);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = MatSetPreallocationCOO(mat_, nnz, coo_rows.data(), coo_cols.data());
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = MatSetValuesCOO(mat_, elm_mass_dev.data(), INSERT_VALUES);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

OmegaHMassIntegrator::~OmegaHMassIntegrator()
{
  if (mat_) {
    MatDestroy(&mat_);
  }
}

Mat OmegaHMassIntegrator::GetMatrix() const noexcept
{
  return mat_;
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

std::unique_ptr<BilinearFormIntegrator> BuildOmegaHMassIntegrator(
  const FunctionSpace& target_space)
{
  return std::make_unique<OmegaHMassIntegrator>(target_space);
}

} // namespace pcms
