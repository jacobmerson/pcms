#ifndef PCMS_BACKENDS_OMEGA_H_SIMPLEX_QUADRATURE_HPP
#define PCMS_BACKENDS_OMEGA_H_SIMPLEX_QUADRATURE_HPP

// Private: reference quadrature rules of simplices, and the element measures
// and barycentric maps the quadratures of an Omega_h mesh (CreateQuadrature,
// CreateIntersectionQuadrature) are built from.

#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/configuration.hpp"
#include "pcms/geometry/detail/geometry.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include "pcms/utility/types.hpp"
#ifdef PCMS_ENABLE_OMEGA_H_MESHFIELDS
#include <MeshField_Integrate.hpp>
#include <MeshField_Shape.hpp>
#include <MeshField_Config.hpp>
#endif
#include <Omega_h_shape.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace pcms::omega_h::detail
{

// Checks that `geometry` is on a 2D or 3D simplex mesh. The quadratures take
// the cell measures from the mesh coordinates and scale them by the volume
// factor of the geometry's coordinate system.
inline void CheckSimplexMesh(const OmegaHGeometry& geometry,
                             const char* context)
{
  const Omega_h::Mesh& mesh = geometry.GetMesh();
  const std::string prefix = std::string(context) + ": ";
  if (mesh.dim() != 2 && mesh.dim() != 3) {
    throw pcms_error(prefix + "the mesh must be 2D or 3D");
  }
  if (mesh.family() != OMEGA_H_SIMPLEX) {
    throw pcms_error(prefix +
                     "the mesh must be a simplex (triangle/tetrahedron) mesh");
  }
}

// Measure of the element of each of `num_points` points, which lie in the
// elements `owners` names, on an affine simplex geometry: |det J| / Dim! from
// the geometry's cell Jacobians.
inline Kokkos::View<Real*, DeviceMemorySpace> ElementMeasuresAtPoints(
  const Geometry& geometry, LO num_points, const OwnerHints& owners,
  const char* context)
{
  const auto* omega_h_geometry = dynamic_cast<const OmegaHGeometry*>(&geometry);
  if (omega_h_geometry == nullptr) {
    throw pcms_error(std::string(context) +
                     ": element measures need an Omega_h geometry");
  }
  PCMS_ALWAYS_ASSERT(owners.entities.extent(0) ==
                     static_cast<size_t>(num_points));
  const int dim = omega_h_geometry->GetMesh().dim();
  const Real reference_measure = dim == 3 ? 1.0 / 6.0 : 0.5;
  const auto determinant = omega_h_geometry->CellJacobian().determinant;
  const auto element_ids = owners.entities;
  Kokkos::View<Real*, DeviceMemorySpace> measures(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "element_measures"),
    num_points);
  Kokkos::parallel_for(
    "ElementMeasuresAtPoints",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, num_points),
    KOKKOS_LAMBDA(const LO i) {
      measures(i) =
        Kokkos::fabs(determinant(element_ids(i))) * reference_measure;
    });
  return measures;
}

// Map barycentric coordinates on a simplex (Dim+1 barycentric components) to
// the global point, given the simplex's Dim+1 vertex coordinates. The
// intersection uses it on the sub-simplices r3d produces.
template <int Dim>
[[nodiscard]] OMEGA_H_INLINE Omega_h::Vector<Dim> GlobalFromBarycentric(
  const Omega_h::Vector<Dim + 1>& barycentric_coord,
  const Omega_h::Few<Omega_h::Vector<Dim>, Dim + 1>& verts_coord)
{
  Omega_h::Vector<Dim> real_coords;
  for (int d = 0; d < Dim; ++d) {
    real_coords[d] = 0.0;
  }
  for (int i = 0; i < Dim + 1; ++i) {
    for (int d = 0; d < Dim; ++d) {
      real_coords[d] += barycentric_coord[i] * verts_coord[i][d];
    }
  }
  return real_coords;
}

#ifdef PCMS_ENABLE_OMEGA_H_MESHFIELDS
// Maps spatial dimension to the MeshField simplex topology whose reference
// quadrature rules we use (triangle in 2D, tetrahedron in 3D).
template <int Dim>
struct SimplexTopology;
template <>
struct SimplexTopology<2>
{
  static constexpr MeshField::Mesh_Topology value = MeshField::Triangle;
};
template <>
struct SimplexTopology<3>
{
  static constexpr MeshField::Mesh_Topology value = MeshField::Tetrahedron;
};

// Barycentric integration points and weights for a reference simplex (triangle
// in 2D, tetrahedron in 3D), taken from MeshField's predefined quadrature rules
// and staged on device for use in element integration kernels.
//
// MeshField::getIntegrationPoints returns a host std::vector, which cannot be
// dereferenced inside a device kernel, so the (tiny) rule is copied into device
// Kokkos views once at construction. Each barycentric point has Dim+1
// components.
//
// The quadrature order is a runtime argument because the required polynomial
// accuracy depends on the source and target element orders (degree =
// source_order + target_order), which are only known at construction.
template <int Dim>
struct IntegrationData
{
  Kokkos::View<Omega_h::Real * [Dim + 1]>
    bary_coords;                        // barycentric coordinates
  Kokkos::View<Omega_h::Real*> weights; // quadrature weights

  explicit IntegrationData(int order)
  {
    // MeshFields has no rule of an order above its accuracies (2 today); it
    // hands back no rule for those
    const auto* rule =
      MeshField::getIntegration<SimplexTopology<Dim>::value>()->getAccurate(
        order);
    if (rule == nullptr) {
      throw pcms_error("IntegrationData: MeshFields has no simplex rule of "
                       "order " +
                       std::to_string(order));
    }
    auto ip_vec = rule->getPoints();
    const std::size_t num_ip = ip_vec.size();

    bary_coords =
      Kokkos::View<Omega_h::Real * [Dim + 1]>("bary_coords", num_ip);
    weights = Kokkos::View<Omega_h::Real*>("weights", num_ip);

    auto bary_coords_host = Kokkos::create_mirror_view(bary_coords);
    auto weights_host = Kokkos::create_mirror_view(weights);
    for (std::size_t i = 0; i < num_ip; ++i) {
      // MeshField returns points in reduced parametric coordinates: only the
      // first Dim barycentric components are stored, with the last implied by
      // the partition of unity. Expand to the full Dim+1 barycentric form
      // consumed by GlobalFromBarycentric.
      Omega_h::Real last = 1.0;
      for (int d = 0; d < Dim; ++d) {
        const Omega_h::Real xi = ip_vec[i].param[d];
        bary_coords_host(i, d) = xi;
        last -= xi;
      }
      bary_coords_host(i, Dim) = last;
      weights_host(i) = ip_vec[i].weight;
    }
    Kokkos::deep_copy(bary_coords, bary_coords_host);
    Kokkos::deep_copy(weights, weights_host);
  }

  int size() const { return bary_coords.extent(0); }
};

#endif // PCMS_ENABLE_OMEGA_H_MESHFIELDS

} // namespace pcms::omega_h::detail

#endif // PCMS_BACKENDS_OMEGA_H_SIMPLEX_QUADRATURE_HPP
