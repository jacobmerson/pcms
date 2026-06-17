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

inline void CheckOmegaHScalarP1Layout(
  CoordinateSystem coordinate_system,
  const std::shared_ptr<const OmegaHLagrangeLayout>& layout,
  const char* context,
  const char* role);

inline void CheckOmegaHScalarP1Layout(
  const FunctionSpace& space,
  const std::shared_ptr<const OmegaHLagrangeLayout>& layout,
  const char* context,
  const char* role)
{
  CheckOmegaHScalarP1Layout(space.GetCoordinateSystem(), layout, context, role);
}

inline void CheckOmegaHScalarP1Layout(
  CoordinateSystem coordinate_system,
  const std::shared_ptr<const OmegaHLagrangeLayout>& layout,
  const char* context,
  const char* role)
{
  if (layout == nullptr) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must use OmegaHLagrangeLayout");
  }
  if (layout->GetOrder() != 1) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must be order-1");
  }
  if (layout->GetNumComponents() != 1) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must have exactly one component");
  }
  if (coordinate_system != CoordinateSystem::Cartesian) {
    throw pcms_error(std::string(context) + ": " + role +
                     " space must use Cartesian coordinates");
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
  r3d::Polytope<2>& poly,
  const double tol = 1e-12)
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
  const int elm,
  const IntersectionResults& intersection,
  const Omega_h::Reals& tgt_coords,
  const Omega_h::Reals& src_coords,
  const Omega_h::LOs& tgt_faces2nodes,
  const Omega_h::LOs& src_faces2nodes,
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
// @tparam order Quadrature order (number of points and polynomial accuracy).
template <int order>
struct IntegrationData
{
  Kokkos::View<MeshField::Vector3*> bary_coords; // barycentric coordinates
  Kokkos::View<Omega_h::Real*> weights;          // quadrature weights

  IntegrationData()
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

} // namespace pcms::detail

#endif // PCMS_TRANSFER_OMEGA_H_FORM_INTEGRATOR_UTILS_HPP
