#include "pcms/transfer/omega_h_rhs_integrator.hpp"
#include "pcms/transfer/omega_h_form_integrator_utils.hpp"
#include "pcms/transfer/petsc_utils.hpp"
#include "pcms/utility/assert.h"
#include <Omega_h_for.hpp>
#include <Omega_h_shape.hpp>
#include <petscksp.h>

namespace pcms
{

// ---------------------------------------------------------------------------
// OmegaHRHSIntegrationPointSet
// ---------------------------------------------------------------------------

OmegaHRHSIntegrationPointSet::OmegaHRHSIntegrationPointSet(
  Kokkos::View<Real**, DeviceMemorySpace> coords)
  : coords_view_(std::move(coords))
{
}

CoordinateView<DeviceMemorySpace>
OmegaHRHSIntegrationPointSet::GetCoordinates() const
{
  return CoordinateView<DeviceMemorySpace>(CoordinateSystem::Cartesian,
                                           MakeConstRank2View(coords_view_));
}

// ---------------------------------------------------------------------------
// OmegaHRHSIntegrator::BuildData
// ---------------------------------------------------------------------------

OmegaHRHSIntegrator::Data OmegaHRHSIntegrator::BuildData(
  const FunctionSpace& source_space, const FunctionSpace& target_space)
{
  return BuildData(
    std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
      source_space.GetLayout()),
    source_space.GetCoordinateSystem(),
    std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
      target_space.GetLayout()),
    target_space.GetCoordinateSystem());
}

OmegaHRHSIntegrator::Data OmegaHRHSIntegrator::BuildData(
  std::shared_ptr<const OmegaHLagrangeLayout> source_layout,
  CoordinateSystem source_coordinate_system,
  std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
  CoordinateSystem target_coordinate_system)
{
  detail::CheckOmegaHScalarP1Layout(source_coordinate_system, source_layout,
                                    "OmegaHRHSIntegrator", "source");
  detail::CheckOmegaHScalarP1Layout(target_coordinate_system, target_layout,
                                    "OmegaHRHSIntegrator", "target");

  Omega_h::Mesh& source_mesh = source_layout->GetMesh();
  Omega_h::Mesh& target_mesh = target_layout->GetMesh();

  PCMS_ALWAYS_ASSERT(source_mesh.dim() == 2);
  PCMS_ALWAYS_ASSERT(target_mesh.dim() == 2);
  PCMS_ALWAYS_ASSERT(source_mesh.family() == OMEGA_H_SIMPLEX);
  PCMS_ALWAYS_ASSERT(target_mesh.family() == OMEGA_H_SIMPLEX);

  const auto intersections = intersectTargets(source_mesh, target_mesh);

  const auto& tgt_coords = target_mesh.coords();
  const auto& tgt_faces2nodes =
    target_mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;
  const auto& src_coords = source_mesh.coords();
  const auto& src_faces2nodes =
    source_mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;

  detail::IntegrationData<2> ip_data;
  const int npts = ip_data.size();
  auto bary_coords = ip_data.bary_coords; // device view
  auto weights = ip_data.weights;         // device view

  const auto device_gids = target_layout->GetGids();
  const GO* gids_ptr = device_gids.data_handle();

  const Omega_h::LOs tgt2src_offsets = intersections.tgt2src_offsets;
  const Omega_h::LOs tgt2src_indices = intersections.tgt2src_indices;
  const int nelems = target_mesh.nelems();

  // Pass 1: count integration points per target element.
  Omega_h::Write<Omega_h::LO> ip_counts(nelems, 0, "rhs_ip_counts");
  Kokkos::parallel_for(
    "rhs_count", nelems, KOKKOS_LAMBDA(int elm) {
      int count = 0;
      detail::ForEachIntersectionSubtriangle(
        elm, {tgt2src_offsets, tgt2src_indices}, tgt_coords, src_coords,
        tgt_faces2nodes, src_faces2nodes,
        [&](const Omega_h::Few<Omega_h::Vector<2>, 3>&,
            const r3d::Few<r3d::Vector<2>, 3>&,
            const r3d::Few<r3d::Vector<2>, 3>&, int,
            Omega_h::Real) { count += npts; });
      ip_counts[elm] = count;
    });
  Kokkos::fence();

  const auto ip_offsets = Omega_h::offset_scan(
    Omega_h::Read<Omega_h::LO>(ip_counts), "rhs_ip_offsets");
  const int num_pts = static_cast<int>(ip_offsets.last());

  // Pass 2: fill coords, node_gids, and coeffs on device.
  Kokkos::View<Real**, DeviceMemorySpace> coords("rhs_coords", num_pts, 2);
  Kokkos::View<PetscInt*, DeviceMemorySpace> node_gids("rhs_node_gids",
                                                       num_pts * 3);
  Kokkos::View<Real*, DeviceMemorySpace> coeffs("rhs_coeffs", num_pts * 3);

  Kokkos::parallel_for(
    "rhs_fill", nelems, KOKKOS_LAMBDA(int elm) {
      const auto tgt_verts = Omega_h::gather_verts<3>(tgt_faces2nodes, elm);
      const Omega_h::Matrix<2, 3> tgt_vert_mat =
        Omega_h::gather_vectors<3, 2>(tgt_coords, tgt_verts);
      Omega_h::Few<Omega_h::Vector<2>, 3> tgt_omh;
      for (int i = 0; i < 3; ++i)
        tgt_omh[i] = {tgt_vert_mat[i][0], tgt_vert_mat[i][1]};

      int ip_local = 0;
      const int offset = ip_offsets[elm];

      detail::ForEachIntersectionSubtriangle(
        elm, {tgt2src_offsets, tgt2src_indices}, tgt_coords, src_coords,
        tgt_faces2nodes, src_faces2nodes,
        [&](const Omega_h::Few<Omega_h::Vector<2>, 3>& tri,
            const r3d::Few<r3d::Vector<2>, 3>&,
            const r3d::Few<r3d::Vector<2>, 3>&, int, Omega_h::Real area) {
          for (int ip_idx = 0; ip_idx < npts; ++ip_idx) {
            const auto bary = bary_coords(ip_idx);
            const double w = weights(ip_idx);
            const auto pt = detail::GlobalFromBarycentric(bary, tri);
            const auto tgt_bary =
              Omega_h::barycentric_from_global<2, 2>(pt, tgt_omh);

            const int global_ip = offset + ip_local;
            coords(global_ip, 0) = pt[0];
            coords(global_ip, 1) = pt[1];
            for (int k = 0; k < 3; ++k) {
              node_gids(global_ip * 3 + k) =
                static_cast<PetscInt>(gids_ptr[tgt_verts[k]]);
              coeffs(global_ip * 3 + k) = tgt_bary[k] * w * 2.0 * area;
            }
            ++ip_local;
          }
        });
    });
  Kokkos::fence();

  Data d;
  d.coords = std::move(coords);
  d.node_gids = std::move(node_gids);
  d.coeffs = std::move(coeffs);
  d.nverts = static_cast<PetscInt>(target_mesh.nverts());
  return d;
}

// ---------------------------------------------------------------------------
// OmegaHRHSIntegrator constructors
// ---------------------------------------------------------------------------

OmegaHRHSIntegrator::OmegaHRHSIntegrator(const FunctionSpace& source_space,
                                         const FunctionSpace& target_space)
  : OmegaHRHSIntegrator(
      std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
        source_space.GetLayout()),
      source_space.GetCoordinateSystem(),
      std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
        target_space.GetLayout()),
      target_space.GetCoordinateSystem())
{
}

OmegaHRHSIntegrator::OmegaHRHSIntegrator(
  std::shared_ptr<const OmegaHLagrangeLayout> source_layout,
  CoordinateSystem source_coordinate_system,
  std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
  CoordinateSystem target_coordinate_system)
  : OmegaHRHSIntegrator(BuildData(std::move(source_layout),
                                   source_coordinate_system,
                                   std::move(target_layout),
                                   target_coordinate_system))
{
}

OmegaHRHSIntegrator::OmegaHRHSIntegrator(Data data)
  : point_set_(std::move(data.coords)),
    node_gids_(std::move(data.node_gids)),
    coeffs_(std::move(data.coeffs))
{
  const PetscInt nnz = static_cast<PetscInt>(node_gids_.extent(0));
  PetscErrorCode ierr = createSeqVec(PETSC_COMM_WORLD, data.nverts, &vec_);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecSetPreallocationCOO(vec_, nnz, node_gids_.data());
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

OmegaHRHSIntegrator::~OmegaHRHSIntegrator()
{
  if (vec_) {
    VecDestroy(&vec_);
  }
}

// ---------------------------------------------------------------------------
// OmegaHRHSIntegrator public interface
// ---------------------------------------------------------------------------

const IntegrationPointSet<DeviceMemorySpace>&
OmegaHRHSIntegrator::GetIntegrationPoints() const noexcept
{
  return point_set_;
}

Vec OmegaHRHSIntegrator::GetVector() const noexcept
{
  return vec_;
}

void OmegaHRHSIntegrator::Assemble(
  Rank2View<const Real, DeviceMemorySpace> sampled_values)
{
  const std::size_t num_pts = static_cast<std::size_t>(node_gids_.extent(0) / 3);
  PCMS_ALWAYS_ASSERT(
    static_cast<std::size_t>(sampled_values.extent(0)) == num_pts);
  PCMS_ALWAYS_ASSERT(sampled_values.extent(1) >= 1);

  VecZeroEntries(vec_);

  auto sv = Kokkos::View<const Real**, Kokkos::LayoutRight, DeviceMemorySpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
    sampled_values.data_handle(), sampled_values.extent(0),
    sampled_values.extent(1));
  Kokkos::View<PetscScalar*, DeviceMemorySpace> coo_vals("rhs_coo_vals",
                                                         num_pts * 3);
  auto coeffs = coeffs_;
  Kokkos::parallel_for(
    "rhs_coo_vals", static_cast<int>(num_pts), KOKKOS_LAMBDA(int i) {
      const PetscScalar f = static_cast<PetscScalar>(sv(i, 0));
      for (int j = 0; j < 3; ++j) {
        coo_vals(i * 3 + j) = static_cast<PetscScalar>(coeffs(i * 3 + j)) * f;
      }
    });

  VecSetValuesCOO(vec_, coo_vals.data(), ADD_VALUES);
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

std::unique_ptr<LinearFormIntegrator> BuildOmegaHConservativeRHSIntegrator(
  const FunctionSpace& source_space, const FunctionSpace& target_space)
{
  return std::make_unique<OmegaHRHSIntegrator>(source_space, target_space);
}

} // namespace pcms
