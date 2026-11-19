#include "pcms/transfer/omega_h_intersection_quadrature.hpp"
#include "pcms/transfer/omega_h_form_integrator_utils.hpp"
#include "pcms/utility/arrays.hpp"
#include <Omega_h_for.hpp>
#include <Omega_h_shape.hpp>

namespace pcms::omega_h
{

namespace
{

template <int Dim>
Quadrature CreateIntersectionQuadratureImpl(
  std::shared_ptr<const OmegaHGeometry> source_geometry,
  std::shared_ptr<const OmegaHGeometry> target_geometry, int quad_order)
{
  // Reference-to-physical Jacobian factor for a simplex: the reference simplex
  // measure is 1/Dim! (1/2 in 2D, 1/6 in 3D), so a physical sub-simplex of
  // measure `m` scales the reference quadrature weights by Dim! * m.
  constexpr Omega_h::Real ref_factor = (Dim == 3) ? 6.0 : 2.0;

  Omega_h::Mesh& source_mesh = source_geometry->GetMesh();
  Omega_h::Mesh& target_mesh = target_geometry->GetMesh();

  const auto intersections = intersectTargets(*source_geometry, target_mesh);

  const auto& tgt_coords = target_mesh.coords();
  const auto& tgt_elems2nodes = target_mesh.ask_down(Dim, Omega_h::VERT).ab2b;
  const auto& src_coords = source_mesh.coords();
  const auto& src_elems2nodes = source_mesh.ask_down(Dim, Omega_h::VERT).ab2b;

  transfer::detail::IntegrationData<Dim> ip_data(quad_order);
  const int npts = ip_data.size();
  auto bary_coords = ip_data.bary_coords; // device view
  auto rule_weights = ip_data.weights;    // device view

  const Omega_h::LOs tgt2src_offsets = intersections.tgt2src_offsets;
  const Omega_h::LOs tgt2src_indices = intersections.tgt2src_indices;
  const int nelems = target_mesh.nelems();

  // Pass 1: count integration points per target element.
  Omega_h::Write<Omega_h::LO> ip_counts(nelems, 0, "rhs_ip_counts");
  Kokkos::parallel_for(
    "quadrature_count", nelems, KOKKOS_LAMBDA(int elm) {
      int count = 0;
      transfer::detail::ForEachIntersectionSubsimplex<Dim>(
        elm, {tgt2src_offsets, tgt2src_indices}, tgt_coords, src_coords,
        tgt_elems2nodes, src_elems2nodes,
        [&](const Omega_h::Few<Omega_h::Vector<Dim>, Dim + 1>&, int,
            Omega_h::Real) { count += npts; });
      ip_counts[elm] = count;
    });
  Kokkos::fence();

  const auto ip_offsets = Omega_h::offset_scan(
    Omega_h::Read<Omega_h::LO>(ip_counts), "rhs_ip_offsets");
  const int num_pts = static_cast<int>(ip_offsets.last());

  // Pass 2: fill the coordinates, target and source element and weight of each
  // integration point on device.
  Kokkos::View<Real**, DeviceMemorySpace> coords("quadrature_coords", num_pts,
                                                 Dim);
  Kokkos::View<LO*, DeviceMemorySpace> target_elements("quadrature_target_elems",
                                                       num_pts);
  Kokkos::View<LO*, DeviceMemorySpace> source_elements("quadrature_source_elems",
                                                       num_pts);
  Kokkos::View<Real*, DeviceMemorySpace> weights("quadrature_weights", num_pts);

  Kokkos::parallel_for(
    "quadrature_fill", nelems, KOKKOS_LAMBDA(int elm) {
      int ip_local = 0;
      const int offset = ip_offsets[elm];

      transfer::detail::ForEachIntersectionSubsimplex<Dim>(
        elm, {tgt2src_offsets, tgt2src_indices}, tgt_coords, src_coords,
        tgt_elems2nodes, src_elems2nodes,
        [&](const Omega_h::Few<Omega_h::Vector<Dim>, Dim + 1>& sub, int src_elm,
            Omega_h::Real measure) {
          for (int ip_idx = 0; ip_idx < npts; ++ip_idx) {
            Omega_h::Vector<Dim + 1> bary;
            for (int d = 0; d < Dim + 1; ++d) {
              bary[d] = bary_coords(ip_idx, d);
            }
            const auto pt =
              transfer::detail::GlobalFromBarycentric<Dim>(bary, sub);
            const int global_ip = offset + ip_local;
            for (int d = 0; d < Dim; ++d) {
              coords(global_ip, d) = pt[d];
            }
            target_elements(global_ip) = elm;
            source_elements(global_ip) = src_elm;
            weights(global_ip) = rule_weights(ip_idx) * ref_factor * measure;
            ++ip_local;
          }
        });
    });
  Kokkos::fence();

  // Both bindings are by the elements the points were generated in: no
  // search on either mesh.
  const CoordinateView<DeviceMemorySpace> points(
    target_geometry->GetCoordinateSystem(), MakeConstRank2View(coords));
  return Quadrature{
    target_geometry->Bind(
      points, OwnerHints{target_geometry, Dim, target_elements}),
    weights, OwnerHints{source_geometry, Dim, source_elements}};
}

} // namespace

Quadrature CreateIntersectionQuadrature(
  std::shared_ptr<const OmegaHGeometry> source,
  std::shared_ptr<const OmegaHGeometry> target, const QuadratureOptions& options)
{
  if (source == nullptr || target == nullptr) {
    throw pcms_error("CreateIntersectionQuadrature: the geometries must not "
                     "be null");
  }
  detail::CheckCartesianSimplexMesh(*source, "CreateIntersectionQuadrature");
  detail::CheckCartesianSimplexMesh(*target, "CreateIntersectionQuadrature");
  const int dim = target->GetMesh().dim();
  if (source->GetMesh().dim() != dim) {
    throw pcms_error("CreateIntersectionQuadrature: source and target mesh "
                     "dimensions differ");
  }
  // a 1-point floor so a P0->P0 pair still gets a valid rule
  const int order = std::max(1, options.order);
  if (dim == 3) {
    return CreateIntersectionQuadratureImpl<3>(std::move(source),
                                               std::move(target), order);
  }
  return CreateIntersectionQuadratureImpl<2>(std::move(source),
                                             std::move(target), order);
}

} // namespace pcms::omega_h
