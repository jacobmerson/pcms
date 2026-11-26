#ifndef PCMS_BACKENDS_OMEGA_H_MESH_INTERSECTION_HPP
#define PCMS_BACKENDS_OMEGA_H_MESH_INTERSECTION_HPP

#include <pcms/configuration.hpp>
#include <pcms/backends/omega_h/omega_h_geometry.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_int_scan.hpp>
// r3d's kernels are called from Kokkos kernels: R3D_USE_KOKKOS makes them
// KOKKOS_INLINE_FUNCTION. Defined here instead of as a public compile
// definition of pcms::transfer; this is the only header that includes r3d.
#ifndef R3D_USE_KOKKOS
#define R3D_USE_KOKKOS
#endif
#include <r3d.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_for.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace pcms::omega_h
{
// Gather the vertex coordinates of a simplex element (triangle for Dim==2,
// tetrahedron for Dim==3) into an r3d simplex, ready for
// r3d::intersect_simplices.
template <int Dim>
[[nodiscard]] OMEGA_H_INLINE r3d::Few<r3d::Vector<Dim>, Dim + 1>
get_vert_coords_of_elem(const Omega_h::Reals& coords,
                        const Omega_h::LOs& elems2nodes, const int id)
{
  const auto elm_verts = Omega_h::gather_verts<Dim + 1>(elems2nodes, id);

  const Omega_h::Matrix<Dim, Dim + 1> elm_vert_coords =
    Omega_h::gather_vectors<Dim + 1, Dim>(coords, elm_verts);

  r3d::Few<r3d::Vector<Dim>, Dim + 1> r3d_vector;
  for (int i = 0; i < Dim + 1; ++i) {
    for (int d = 0; d < Dim; ++d) {
      r3d_vector[i][d] = elm_vert_coords[i][d];
    }
  }

  return r3d_vector;
}

/**
 * @brief Conservative test for a zero-volume overlap between two simplices.
 *
 * If verticies of one simplex lies in the closed outer half-space of some face
 * plane of the other, their intersection is contained in that plane and so has
 * no volume.
 *
 * @return true when the overlap is provably degenerate and the clip can be
 * skipped
 */
template <int Dim>
[[nodiscard]] OMEGA_H_INLINE bool simplices_have_degenerate_overlap(
  const r3d::Few<r3d::Vector<Dim>, Dim + 1>& a,
  const r3d::Few<r3d::Vector<Dim>, Dim + 1>& b)
{
  // Each iteration treats one simplex as the half-space owner. Both are
  // needed: a face of `a` may separate the pair while no face of `b` does.
  for (int side = 0; side < 2; ++side) {
    const auto& owner = (side == 0) ? a : b;
    const auto& other = (side == 0) ? b : a;

    // Face `f` is the face opposite vertex `f`.
    for (int f = 0; f < Dim + 1; ++f) {
      // Vertices spanning the face, in index order with `f` skipped.
      int face_verts[Dim];
      int n_face_verts = 0;
      for (int v = 0; v < Dim + 1; ++v) {
        if (v != f) {
          face_verts[n_face_verts++] = v;
        }
      }
      const auto& origin = owner[face_verts[0]];

      // Face normal, oriented to point away from the opposite vertex so the
      // owner's interior lies on the negative side.
      r3d::Vector<Dim> normal;
      if constexpr (Dim == 2) {
        const auto edge = owner[face_verts[1]] - origin;
        normal[0] = edge[1];
        normal[1] = -edge[0];
      } else {
        const auto e1 = owner[face_verts[1]] - origin;
        const auto e2 = owner[face_verts[2]] - origin;
        normal[0] = e1[1] * e2[2] - e1[2] * e2[1];
        normal[1] = e1[2] * e2[0] - e1[0] * e2[2];
        normal[2] = e1[0] * e2[1] - e1[1] * e2[0];
      }

      Omega_h::Real opposite_side = 0.0;
      Omega_h::Real normal_sq = 0.0;
      for (int d = 0; d < Dim; ++d) {
        opposite_side += normal[d] * (owner[f][d] - origin[d]);
        normal_sq += normal[d] * normal[d];
      }
      // A degenerate face gives no usable axis.
      if (normal_sq == 0.0) {
        continue;
      }
      const Omega_h::Real orientation = (opposite_side > 0.0) ? -1.0 : 1.0;

      bool all_outside = true;
      for (int v = 0; v < Dim + 1 && all_outside; ++v) {
        Omega_h::Real signed_distance = 0.0;
        for (int d = 0; d < Dim; ++d) {
          signed_distance += normal[d] * (other[v][d] - origin[d]);
        }
        all_outside = (orientation * signed_distance) >= 0.0;
      }
      if (all_outside) {
        return true;
      }
    }
  }
  return false;
}

/**
 * @brief Stores results of mesh element intersections for conservative
 * transfer.
 *
 * Contains mappings from each target element to the list of source elements
 * that intersect with it. Used to guide integration over overlapping regions.
 *
 * - `tgt2src_offsets[i]` is the offset into `tgt2src_indices` where source
 *    elements for target element `i` begin.
 * - `tgt2src_indices` contains flattened indices of source elements per target.
 */
struct IntersectionResults
{
  Omega_h::LOs tgt2src_offsets;
  Omega_h::LOs tgt2src_indices;
};

/**
 * @brief Computes source-target element intersections for conservative
 * projection.
 *
 * For each target element in the target mesh, this function identifies source
 * elements from the source mesh that geometrically intersect with it: the
 * candidates of its bounding box on a candidate map of the source elements,
 * in ascending order. The result is returned as a compact mapping.
 *
 * @param source geometry of the source mesh
 * @param target_mesh The target Omega_h mesh.
 * @param use_prefilter if true, skip degenerate clips
 * @return An IntersectionResults struct containing target-to-source mapping
 * data.
 * @throws pcms_error if the source geometry's mesh and the target mesh differ
 * in dimension
 *
 * @note The intersection test is done using 2D polygon intersection routines
 * from r3d. Only valid (non-degenerate) polygonal intersections are included.
 *
 */
IntersectionResults intersectTargets(const OmegaHGeometry& source,
                                     Omega_h::Mesh& target_mesh,
                                     bool use_prefilter = true);
} // namespace pcms::omega_h
#endif // PCMS_BACKENDS_OMEGA_H_MESH_INTERSECTION_HPP
