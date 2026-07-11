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
#include <Omega_h_shape.hpp>
#include <petscksp.h>

namespace pcms
{

OmegaHMassIntegrator::OmegaHMassIntegrator(const FunctionSpace& target_space)
  : OmegaHMassIntegrator(std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
                           target_space.GetLayout()),
                         target_space.GetCoordinateSystem())
{
}

OmegaHMassIntegrator::OmegaHMassIntegrator(
  std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
  CoordinateSystem coordinate_system)
{
  detail::CheckOmegaHScalarLagrangeLayout(coordinate_system, target_layout,
                                          "OmegaHMassIntegrator", "target");

  Omega_h::Mesh& mesh = target_layout->GetMesh();
  const auto device_gids = target_layout->GetGids();
  const GO* gids_ptr = device_gids.data_handle();
  const PetscInt num_dofs =
    static_cast<PetscInt>(target_layout->GetNumGlobalDofHolder());
  const int nelems = mesh.nelems();

  if (target_layout->GetOrder() == 0) {
    // P0 target: piecewise-constant basis functions have disjoint support, so
    // the mass matrix is diagonal with M_ee = area(e). One COO entry per
    // element on its own (element-id) diagonal.
    const auto& coords = mesh.coords();
    const auto& faces2nodes = mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;

    const PetscInt nnz = static_cast<PetscInt>(nelems);
    Kokkos::View<PetscInt*, DeviceMemorySpace> coo_rows("mass_coo_rows", nnz);
    Kokkos::View<PetscInt*, DeviceMemorySpace> coo_cols("mass_coo_cols", nnz);
    Kokkos::View<PetscScalar*, DeviceMemorySpace> vals("mass_vals", nnz);
    Kokkos::parallel_for(
      "mass_p0_diag", nelems, KOKKOS_LAMBDA(int e) {
        const auto verts = Omega_h::gather_verts<3>(faces2nodes, e);
        const Omega_h::Matrix<2, 3> vm =
          Omega_h::gather_vectors<3, 2>(coords, verts);
        Omega_h::Few<Omega_h::Vector<2>, 2> basis;
        basis[0] = vm[1] - vm[0];
        basis[1] = vm[2] - vm[0];
        const Omega_h::Real area =
          Kokkos::fabs(Omega_h::triangle_area_from_basis(basis));
        const PetscInt g = static_cast<PetscInt>(gids_ptr[e]);
        coo_rows(e) = g;
        coo_cols(e) = g;
        vals(e) = static_cast<PetscScalar>(area);
      });

    PetscErrorCode ierr =
      createSeqAIJMat(PETSC_COMM_WORLD, num_dofs, num_dofs, 0, nullptr, &mat_);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetPreallocationCOO(mat_, nnz, coo_rows.data(), coo_cols.data());
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = MatSetValuesCOO(mat_, vals.data(), INSERT_VALUES);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    return;
  }

  // P1 target: consistent mass matrix assembled from MeshField per-element 3x3
  // blocks. (Higher MeshField orders extend this branch via
  // getTriangleElement.)
  MeshField::OmegahMeshField<DefaultExecutionSpace, 2,
                             MeshField::KokkosController>
    omf(mesh);
  auto coordField = omf.getCoordField();
  const auto [shp, map] = MeshField::Omegah::getTriangleElement<1>(mesh);
  MeshField::FieldElement coordFe(mesh.nelems(), coordField, shp, map);
  auto elm_mass_dev = buildElementMassMatrix(mesh, coordFe);

  // Build COO sparsity pattern on device: each element contributes a 3x3 block.
  const auto& faces2nodes = mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;

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
  PetscErrorCode ierr =
    createSeqAIJMat(PETSC_COMM_WORLD, num_dofs, num_dofs, 0, nullptr, &mat_);
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
