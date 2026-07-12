#ifndef PCMS_TRANSFER_OMEGA_H_FORM_INTEGRATOR_UTILS_HPP
#define PCMS_TRANSFER_OMEGA_H_FORM_INTEGRATOR_UTILS_HPP

#include "pcms/field/function_space.h"
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/transfer/mesh_intersection.hpp"
#include "pcms/utility/assert.h"
#include <MeshField_Integrate.hpp>
#include <MeshField_Shape.hpp>
#include <Omega_h_shape.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace pcms::detail
{

// Shared checks for a scalar Cartesian Lagrange space on a 2D simplex mesh,
// independent of order. Order is validated separately by the callers below.
inline void CheckOmegaHScalarSimplex2DLayout(
  CoordinateSystem coordinate_system,
  const std::shared_ptr<const OmegaHLagrangeLayout>& layout,
  const char* context, const char* role)
{
  if (layout == nullptr) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must use OmegaHLagrangeLayout");
  }
  if (layout->GetNumComponents() != 1) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must have exactly one component");
  }
  if (coordinate_system != CoordinateSystem::Cartesian) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must use Cartesian coordinates");
  }
  const Omega_h::Mesh& mesh = layout->GetMesh();
  if (mesh.dim() != 2) {
    throw pcms_error(std::string(context) + ": " + role + " mesh must be 2D");
  }
  if (mesh.family() != OMEGA_H_SIMPLEX) {
    throw pcms_error(std::string(context) + ": " + role +
                     " mesh must be a simplex (triangle) mesh");
  }
}

// Strict order-1 check (used where only P1 is supported, e.g. Monte-Carlo RHS).
inline void CheckOmegaHScalarP1Layout(
  CoordinateSystem coordinate_system,
  const std::shared_ptr<const OmegaHLagrangeLayout>& layout,
  const char* context, const char* role)
{
  CheckOmegaHScalarSimplex2DLayout(coordinate_system, layout, context, role);
  if (layout->GetOrder() != 1) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must be order-1");
  }
}

// Conservative-projection check: any supported Lagrange order (P0 or P1). The
// intersection integrator handles source and target orders independently, so
// this replaces the strict P1 requirement on those paths.
inline void CheckOmegaHScalarLagrangeLayout(
  CoordinateSystem coordinate_system,
  const std::shared_ptr<const OmegaHLagrangeLayout>& layout,
  const char* context, const char* role)
{
  CheckOmegaHScalarSimplex2DLayout(coordinate_system, layout, context, role);
  const int order = layout->GetOrder();
  if (order != 0 && order != 1) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must be order-0 or order-1");
  }
}

[[nodiscard]] OMEGA_H_INLINE Omega_h::Vector<2> GlobalFromBarycentric(
  const MeshField::Vector3& barycentric_coord,
  const Omega_h::Few<Omega_h::Vector<2>, 3>& verts_coord)
{
  Omega_h::Vector<2> real_coords = {0.0, 0.0};
  for (int i = 0; i < 3; ++i) {
    real_coords[0] += barycentric_coord[i] * verts_coord[i][0];
    real_coords[1] += barycentric_coord[i] * verts_coord[i][1];
  }
  return real_coords;
}

[[nodiscard]] OMEGA_H_INLINE Omega_h::Vector<3> EvaluateBarycentric(
  const Omega_h::Vector<2>& point,
  const r3d::Few<r3d::Vector<2>, 3>& verts_coord)
{
  Omega_h::Few<Omega_h::Vector<2>, 3> omegah_vector;
  for (int i = 0; i < 3; ++i) {
    omegah_vector[i][0] = verts_coord[i][0];
    omegah_vector[i][1] = verts_coord[i][1];
  }
  return Omega_h::barycentric_from_global<2, 2>(point, omegah_vector);
}

[[nodiscard]] OMEGA_H_INLINE int RemoveDuplicateVerticesAndFixLinks(
  r3d::Polytope<2>& poly, const double tol = 1e-12)
{
  const int old_n = poly.nverts;
  int new_n = 0;

  for (int i = 0; i < old_n; ++i) {
    const auto& pi = poly.verts[i].pos;
    bool dup = false;
    for (int j = 0; j < new_n; ++j) {
      const auto& pj = poly.verts[j].pos;
      if (Kokkos::fabs(pi[0] - pj[0]) < tol &&
          Kokkos::fabs(pi[1] - pj[1]) < tol) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      poly.verts[new_n] = poly.verts[i];
      ++new_n;
    }
  }
  poly.nverts = new_n;

  if (new_n >= 3) {
    double cx = 0.0;
    double cy = 0.0;
    for (int i = 0; i < new_n; ++i) {
      cx += poly.verts[i].pos[0];
      cy += poly.verts[i].pos[1];
    }
    cx /= new_n;
    cy /= new_n;

    int order[r3d::MaxVerts<2>::value];
    for (int i = 0; i < new_n; ++i) {
      order[i] = i;
    }

    for (int i = 0; i < new_n - 1; ++i) {
      for (int j = i + 1; j < new_n; ++j) {
        const double a1 = Kokkos::atan2(poly.verts[order[i]].pos[1] - cy,
                                        poly.verts[order[i]].pos[0] - cx);
        const double a2 = Kokkos::atan2(poly.verts[order[j]].pos[1] - cy,
                                        poly.verts[order[j]].pos[0] - cx);
        if (a1 > a2) {
          int t = order[i];
          order[i] = order[j];
          order[j] = t;
        }
      }
    }

    r3d::Vertex<2> tmp[r3d::MaxVerts<2>::value];
    for (int i = 0; i < new_n; ++i) {
      tmp[i] = poly.verts[order[i]];
    }

    for (int i = 0; i < new_n; ++i) {
      poly.verts[i] = tmp[i];
    }

    for (int i = 0; i < new_n; ++i) {
      const int prev = (i - 1 + new_n) % new_n;
      const int next = (i + 1) % new_n;
      poly.verts[i].pnbrs[0] = prev;
      poly.verts[i].pnbrs[1] = next;
    }

    double area = r3d::measure(poly);
    if (area < 0.0) {
      for (int i = 0; i < new_n / 2; ++i) {
        r3d::Vertex<2> t = poly.verts[i];
        poly.verts[i] = poly.verts[new_n - 1 - i];
        poly.verts[new_n - 1 - i] = t;
      }
      for (int i = 0; i < new_n; ++i) {
        const int prev = (i - 1 + new_n) % new_n;
        const int next = (i + 1) % new_n;
        poly.verts[i].pnbrs[0] = prev;
        poly.verts[i].pnbrs[1] = next;
      }
    }
  } else {
    for (int i = 0; i < new_n; ++i) {
      poly.verts[i].pnbrs[0] = -1;
      poly.verts[i].pnbrs[1] = -1;
    }
  }

  return new_n;
}

template <typename TriangleOp>
OMEGA_H_INLINE void ForEachIntersectionSubtriangle(
  const int elm, const IntersectionResults& intersection,
  const Omega_h::Reals& tgt_coords, const Omega_h::Reals& src_coords,
  const Omega_h::LOs& tgt_faces2nodes, const Omega_h::LOs& src_faces2nodes,
  TriangleOp&& op)
{
  auto tgt_elm_vert_coords =
    get_vert_coords_of_elem(tgt_coords, tgt_faces2nodes, elm);
  const int start = intersection.tgt2src_offsets[elm];
  const int end = intersection.tgt2src_offsets[elm + 1];

  for (int i = start; i < end; ++i) {
    const int current_src_elm = intersection.tgt2src_indices[i];
    auto src_elm_vert_coords =
      get_vert_coords_of_elem(src_coords, src_faces2nodes, current_src_elm);
    r3d::Polytope<2> poly;
    r3d::intersect_simplices(poly, tgt_elm_vert_coords, src_elm_vert_coords);
    auto nverts = RemoveDuplicateVerticesAndFixLinks(poly, 1e-12);
    auto poly_area = r3d::measure(poly);

    for (int j = 1; j < nverts - 1; ++j) {
      auto& p0 = poly.verts[0].pos;
      auto& p1 = poly.verts[j].pos;
      auto& p2 = poly.verts[j + 1].pos;

      Omega_h::Few<Omega_h::Vector<2>, 3> tri_coords;
      tri_coords[0] = {p0[0], p0[1]};
      tri_coords[1] = {p1[0], p1[1]};
      tri_coords[2] = {p2[0], p2[1]};

      Omega_h::Few<Omega_h::Vector<2>, 2> basis;
      basis[0] = tri_coords[1] - tri_coords[0];
      basis[1] = tri_coords[2] - tri_coords[0];

      Omega_h::Real area =
        Kokkos::fabs(Omega_h::triangle_area_from_basis(basis));

      const double eps_area = abs_tol + rel_tol * poly_area;
      if (area <= eps_area) {
        continue;
      }

      op(tri_coords, tgt_elm_vert_coords, src_elm_vert_coords, current_src_elm,
         area);
    }
  }
}

// Barycentric integration points and weights for a reference triangle, taken
// from MeshField's predefined triangle quadrature rules and staged on device
// for use in element integration kernels.
//
// MeshField::getIntegrationPoints returns a host std::vector, which cannot be
// dereferenced inside a device kernel, so the (tiny) rule is copied into device
// Kokkos views once at construction.
//
// The quadrature order is a runtime argument because the required polynomial
// accuracy depends on the source and target element orders (degree =
// source_order + target_order), which are only known at construction.
struct IntegrationData
{
  Kokkos::View<MeshField::Vector3*> bary_coords; // barycentric coordinates
  Kokkos::View<Omega_h::Real*> weights;          // quadrature weights

  explicit IntegrationData(int order)
  {
    auto ip_vec = MeshField::getIntegrationPoints<MeshField::Triangle>(order);
    const std::size_t num_ip = ip_vec.size();

    bary_coords = Kokkos::View<MeshField::Vector3*>("bary_coords", num_ip);
    weights = Kokkos::View<Omega_h::Real*>("weights", num_ip);

    auto bary_coords_host = Kokkos::create_mirror_view(bary_coords);
    auto weights_host = Kokkos::create_mirror_view(weights);
    for (std::size_t i = 0; i < num_ip; ++i) {
      bary_coords_host(i) = ip_vec[i].param;
      weights_host(i) = ip_vec[i].weight;
    }
    Kokkos::deep_copy(bary_coords, bary_coords_host);
    Kokkos::deep_copy(weights, weights_host);
  }

  int size() const { return bary_coords.extent(0); }
};

// Target Lagrange basis on a triangle, parameterized by element order, for the
// conservative-projection RHS assembly. Order 0 is a single element-constant
// DOF; order 1 is the three vertex (barycentric) DOFs. Higher orders slot in as
// additional specializations, kept in lock-step with element_dispatch.h.
//
// Each specialization provides, for a target element `elm` with local vertex
// ids `verts` and vertex coordinates `tgt_verts`:
//   ndof            number of local target DOFs
//   Index(...)      active PETSc row for local dof k
//   Values(pt, ...) basis values at the (global) integration point pt
template <int Order>
struct TargetTriBasis;

template <>
struct TargetTriBasis<0>
{
  static constexpr int ndof = 1;

  template <typename Permutation>
  KOKKOS_INLINE_FUNCTION static LO Index(const Permutation& permutation,
                                         int elm,
                                         const Omega_h::Few<Omega_h::LO, 3>&,
                                         int /*k*/)
  {
    return permutation(elm);
  }

  KOKKOS_INLINE_FUNCTION static void Values(
    const Omega_h::Vector<2>&, const Omega_h::Few<Omega_h::Vector<2>, 3>&,
    Omega_h::Real out[ndof])
  {
    out[0] = 1.0;
  }
};

template <>
struct TargetTriBasis<1>
{
  static constexpr int ndof = 3;

  template <typename Permutation>
  KOKKOS_INLINE_FUNCTION static LO Index(
    const Permutation& permutation, int /*elm*/,
    const Omega_h::Few<Omega_h::LO, 3>& verts, int k)
  {
    return permutation(verts[k]);
  }

  // P1 basis functions are the barycentric coordinates of the target element
  // evaluated at the (global) integration point.
  KOKKOS_INLINE_FUNCTION static void Values(
    const Omega_h::Vector<2>& pt,
    const Omega_h::Few<Omega_h::Vector<2>, 3>& tgt_verts,
    Omega_h::Real out[ndof])
  {
    const auto bary = Omega_h::barycentric_from_global<2, 2>(pt, tgt_verts);
    out[0] = bary[0];
    out[1] = bary[1];
    out[2] = bary[2];
  }
};

} // namespace pcms::detail

#endif // PCMS_TRANSFER_OMEGA_H_FORM_INTEGRATOR_UTILS_HPP
