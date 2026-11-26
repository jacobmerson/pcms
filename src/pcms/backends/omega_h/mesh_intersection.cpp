#include "pcms/backends/omega_h/mesh_intersection.hpp"
#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/backends/omega_h/omega_h.hpp"
#include "pcms/localization/candidate_map.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/mesh_geometry.hpp"

namespace pcms::omega_h
{
namespace
{
// The source elements whose overlap with target element `id` has volume: the
// candidates of its box, ascending; counted, and written to `indices` unless
// null.
template <int Dim>
OMEGA_H_INLINE int Intersections(
  const r3d::Few<r3d::Vector<Dim>, Dim + 1>& target, Omega_h::Real measure,
  const Omega_h::Reals& src_coords, const Omega_h::LOs& src_elems2nodes,
  const Omega_h::Reals& src_measures,
  const typename CandidateMap<Dim>::Bins& candidates, LO id, bool use_prefilter,
  LO* indices)
{
  int count = 0;
  for (auto k = candidates.row_map(id); k < candidates.row_map(id + 1); ++k) {
    const LO source = candidates.entries(k);
    const auto source_coords =
      get_vert_coords_of_elem<Dim>(src_coords, src_elems2nodes, source);
    // a clip of a plane-separated pair has no volume
    if (use_prefilter &&
        simplices_have_degenerate_overlap<Dim>(target, source_coords)) {
      continue;
    }
    r3d::Polytope<Dim> intersection;
    r3d::intersect_simplices(intersection, target, source_coords);
    // r3d::measure is signed by the orientation of the target simplex
    const auto intersected = Kokkos::fabs(r3d::measure(intersection));
    const auto eps = Kokkos::fmax(
      PCMS_INTERSECTION_ABS_TOL,
      PCMS_INTERSECTION_REL_TOL * Kokkos::fmax(measure, src_measures[source]));
    // a polygon in 2D, a polyhedron in 3D
    if (intersection.nverts >= Dim + 1 && intersected >= eps) {
      if (indices != nullptr) {
        indices[count] = source;
      }
      ++count;
    }
  }
  return count;
}

template <int Dim>
IntersectionResults intersectTargetsImpl(const OmegaHGeometry& source,
                                         Omega_h::Mesh& target_mesh,
                                         bool use_prefilter)
{
  auto& source_mesh = source.GetMesh();
  const auto map = CreateCandidateMap(
    MeshSimplices<Dim>(source_mesh.coords().view(),
                       source_mesh.ask_elem_verts().view()),
    CreateUniformGridFromMesh<Dim>(
      source_mesh,
      pcms::detail::DivisionsPerAxisForMesh<Dim>(source_mesh.nelems())));
  const auto targets = MeshSimplices<Dim>(target_mesh.coords().view(),
                                          target_mesh.ask_elem_verts().view());
  const auto nelems_target = target_mesh.nelems();
  Kokkos::View<AABBox<Dim>*> boxes("target_boxes", nelems_target);
  Kokkos::parallel_for(
    "target_boxes", nelems_target,
    KOKKOS_LAMBDA(LO t) { boxes(t) = BoundingBox(targets(t)); });
  const auto candidates = map.GetCandidates(boxes);

  const auto& tgt_coords = target_mesh.coords();
  const auto& src_coords = source_mesh.coords();
  const auto& tgt_elems2nodes = target_mesh.ask_down(Dim, Omega_h::VERT).ab2b;
  const auto& src_elems2nodes = source_mesh.ask_down(Dim, Omega_h::VERT).ab2b;
  const auto& src_measures = measure_elements_real(&source_mesh);
  const auto& tgt_measures = measure_elements_real(&target_mesh);
  const auto intersections = KOKKOS_LAMBDA(LO id, LO * indices)
  {
    return Intersections<Dim>(
      get_vert_coords_of_elem<Dim>(tgt_coords, tgt_elems2nodes, id),
      tgt_measures[id], src_coords, src_elems2nodes, src_measures, candidates,
      id, use_prefilter, indices);
  };
  Omega_h::Write<Omega_h::LO> counts(
    nelems_target, "number of intersections in each target element");
  Omega_h::parallel_for(
    nelems_target, OMEGA_H_LAMBDA(const Omega_h::LO id) {
      counts[id] = intersections(id, nullptr);
    });
  const auto tgt2src_offsets =
    Omega_h::offset_scan(Omega_h::Read(counts), "offsets for intersections");
  Omega_h::Write<Omega_h::LO> tgt2src_indices(
    tgt2src_offsets.last(),
    "indices of the source elements that intersect the given target element");
  Omega_h::parallel_for(
    nelems_target, OMEGA_H_LAMBDA(const Omega_h::LO id) {
      intersections(id, tgt2src_indices.data() + tgt2src_offsets[id]);
    });
  return {.tgt2src_offsets = tgt2src_offsets,
          .tgt2src_indices = Omega_h::read(tgt2src_indices)};
}

} // namespace

IntersectionResults intersectTargets(const OmegaHGeometry& source,
                                     Omega_h::Mesh& target_mesh,
                                     bool use_prefilter)
{
  if (source.GetMesh().dim() != target_mesh.dim()) {
    throw pcms_error("intersectTargets: the source geometry's mesh and the "
                     "target mesh differ in dimension");
  }
  if (target_mesh.dim() == 3) {
    return intersectTargetsImpl<3>(source, target_mesh, use_prefilter);
  }
  return intersectTargetsImpl<2>(source, target_mesh, use_prefilter);
}
} // namespace pcms::omega_h
