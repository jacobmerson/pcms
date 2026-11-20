#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/backends/omega_h/simplex_quadrature.hpp"
#include "pcms/configuration.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include <Kokkos_Random.hpp>
#include <Omega_h_shape.hpp>

namespace pcms::omega_h
{

namespace
{

// Maps a uniform point on the unit square to uniform barycentric coordinates on
// a triangle. Shape Distributions (ACM Transactions on Graphics, Vol. 21,
// No. 4, October 2002.) page 814 Eq 1.
KOKKOS_INLINE_FUNCTION Omega_h::Vector<3> UniformTriangleBarycentric(Real u,
                                                                     Real v)
{
  const Real s = Kokkos::sqrt(u);
  return {1.0 - s, s * (1.0 - v), s * v};
}

// Maps three uniform draws on the unit cube to uniform barycentric coordinates
// on a tetrahedron via the cut-and-fold method (Rocchini & Cignoni,
// "Generating Random Points in a Tetrahedron", J. Graphics Tools 2000).
KOKKOS_INLINE_FUNCTION Omega_h::Vector<4> UniformTetBarycentric(Real s, Real t,
                                                                Real u)
{
  if (s + t > 1.0) { // fold the cube into a prism
    s = 1.0 - s;
    t = 1.0 - t;
  }
  if (t + u > 1.0) { // fold the prism into a tetrahedron
    const Real tmp = u;
    u = 1.0 - s - t;
    t = 1.0 - tmp;
  } else if (s + t + u > 1.0) {
    const Real tmp = u;
    u = s + t + u - 1.0;
    s = 1.0 - t - tmp;
  }
  const Real a = 1.0 - s - t - u;
  return {a, s, t, u};
}

template <int Dim>
KOKKOS_INLINE_FUNCTION Omega_h::Vector<Dim + 1> UniformSimplexBarycentric(
  const Real r[Dim])
{
  if constexpr (Dim == 3) {
    return UniformTetBarycentric(r[0], r[1], r[2]);
  } else {
    return UniformTriangleBarycentric(r[0], r[1]);
  }
}

// Fills the coordinates and element of all samples of one element.
// unit_sample_at(s, r) fills the s-th sample's Dim unit-hypercube draws.
template <int Dim, typename UnitSampleAt>
KOKKOS_INLINE_FUNCTION void FillElementSamples(
  const int elm, const int samples_per_element, const Omega_h::LOs& elems2nodes,
  const Omega_h::Reals& mesh_coords,
  const Kokkos::View<LO*, DeviceMemorySpace>& elements,
  const Kokkos::View<Real**, DeviceMemorySpace>& coords,
  const UnitSampleAt& unit_sample_at)
{
  constexpr int nv = Dim + 1;
  const auto verts = Omega_h::gather_verts<nv>(elems2nodes, elm);
  for (int s = 0; s < samples_per_element; ++s) {
    Real r[Dim];
    unit_sample_at(s, r);
    const auto bary = UniformSimplexBarycentric<Dim>(r);

    const int i = elm * samples_per_element + s;
    elements(i) = elm;
    for (int d = 0; d < Dim; ++d) {
      coords(i, d) = 0;
    }
    for (int k = 0; k < nv; ++k) {
      for (int d = 0; d < Dim; ++d) {
        coords(i, d) += bary[k] * mesh_coords[verts[k] * Dim + d];
      }
    }
  }
}

// Samples samples_per_element from a uniform random distribution over each
// element, writing their coordinates and elements.
template <int Dim>
void FillRandomSamples(int nelems, int samples_per_element,
                       const Omega_h::LOs& elems2nodes,
                       const Omega_h::Reals& mesh_coords,
                       const Kokkos::View<LO*, DeviceMemorySpace>& elements,
                       const Kokkos::View<Real**, DeviceMemorySpace>& coords,
                       uint64_t seed)
{
  Kokkos::Random_XorShift64_Pool<DefaultExecutionSpace> pool(seed);
  Kokkos::parallel_for(
    "quadrature_fill_random",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, nelems),
    KOKKOS_LAMBDA(int elm) {
      auto gen = pool.get_state();
      FillElementSamples<Dim>(elm, samples_per_element, elems2nodes,
                              mesh_coords, elements, coords,
                              [&](int /*s*/, Real r[Dim]) {
                                for (int d = 0; d < Dim; ++d) {
                                  r[d] = gen.drand();
                                }
                              });
      pool.free_state(gen);
    });
  Kokkos::fence();
}

#ifdef PCMS_ENABLE_OMEGA_H_MESHFIELDS
// Fills the coordinates and element of the points of the reference rule on
// each element, and the rule's weight scaled to the reference measure's
// complement (Dim!): times the element measure it is the point's weight.
template <int Dim>
void FillReferenceRule(int nelems, int order, const Omega_h::LOs& elems2nodes,
                       const Omega_h::Reals& mesh_coords,
                       const Kokkos::View<LO*, DeviceMemorySpace>& elements,
                       const Kokkos::View<Real**, DeviceMemorySpace>& coords,
                       const Kokkos::View<Real*, DeviceMemorySpace>& unit)
{
  // the reference simplex has measure 1/Dim!
  constexpr Real RefFactor = (Dim == 3) ? 6.0 : 2.0;
  detail::IntegrationData<Dim> rule(order);
  const int npts = rule.size();
  const auto bary_coords = rule.bary_coords;
  const auto weights = rule.weights;
  Kokkos::parallel_for(
    "quadrature_fill_rule",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, nelems),
    KOKKOS_LAMBDA(int elm) {
      const auto verts = Omega_h::gather_verts<Dim + 1>(elems2nodes, elm);
      for (int q = 0; q < npts; ++q) {
        const int i = elm * npts + q;
        elements(i) = elm;
        unit(i) = weights(q) * RefFactor;
        for (int d = 0; d < Dim; ++d) {
          coords(i, d) = 0;
        }
        for (int k = 0; k < Dim + 1; ++k) {
          for (int d = 0; d < Dim; ++d) {
            coords(i, d) += bary_coords(q, k) * mesh_coords[verts[k] * Dim + d];
          }
        }
      }
    });
  Kokkos::fence();
}
#endif

// The weight of each sample is its unit weight times the measure of its
// element.
void ScaleByMeasures(const Kokkos::View<Real*, DeviceMemorySpace>& unit,
                     const Kokkos::View<const Real*, DeviceMemorySpace>& measures)
{
  Kokkos::parallel_for(
    "quadrature_weights",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(unit.extent(0))),
    KOKKOS_LAMBDA(const LO i) { unit(i) *= measures(i); });
}

} // namespace

Quadrature OmegaHGeometry::CreateQuadrature(
  const QuadratureOptions& options) const
{
  detail::CheckSimplexMesh(*this, "OmegaHGeometry::CreateQuadrature");
  const int dim = mesh_.dim();
  const bool random = options.samples_per_cell > 0;
  if (!random && options.order < 1) {
    throw pcms_error("OmegaHGeometry::CreateQuadrature: the order must be "
                     "positive");
  }
#ifndef PCMS_ENABLE_OMEGA_H_MESHFIELDS
  if (!random) {
    throw pcms_error("OmegaHGeometry::CreateQuadrature: the reference rules "
                     "need the MeshFields backend "
                     "(PCMS_ENABLE_OMEGA_H_MESHFIELDS)");
  }
#endif

  const int nelems = mesh_.nelems();
  const auto elems2nodes = mesh_.ask_down(dim, Omega_h::VERT).ab2b;
  const auto mesh_coords = mesh_.coords();
  int per_cell = options.samples_per_cell;
#ifdef PCMS_ENABLE_OMEGA_H_MESHFIELDS
  if (!random) {
    per_cell = dim == 3
                 ? detail::IntegrationData<3>(options.order).size()
                 : detail::IntegrationData<2>(options.order).size();
  }
#endif
  const auto num_points = static_cast<std::size_t>(nelems) * per_cell;
  Kokkos::View<LO*, DeviceMemorySpace> elements("quadrature_elements",
                                                num_points);
  Kokkos::View<Real**, DeviceMemorySpace> coords("quadrature_coords",
                                                 num_points, dim);
  // per point: the weight divided by its element's measure
  Kokkos::View<Real*, DeviceMemorySpace> weights("quadrature_weights",
                                                 num_points);
  if (random) {
    Kokkos::deep_copy(weights, 1.0 / per_cell);
    if (dim == 3) {
      FillRandomSamples<3>(nelems, per_cell, elems2nodes, mesh_coords,
                           elements, coords, options.seed);
    } else {
      FillRandomSamples<2>(nelems, per_cell, elems2nodes, mesh_coords,
                           elements, coords, options.seed);
    }
  } else {
#ifdef PCMS_ENABLE_OMEGA_H_MESHFIELDS
    if (dim == 3) {
      FillReferenceRule<3>(nelems, options.order, elems2nodes, mesh_coords,
                           elements, coords, weights);
    } else {
      FillReferenceRule<2>(nelems, options.order, elems2nodes, mesh_coords,
                           elements, coords, weights);
    }
#endif
  }

  // The points lie in the elements they were generated in: bind them there
  // without a search, and take the element measures from the geometry's
  // differential at the bound points.
  const OwnerHints owners{shared_from_this(), dim, elements};
  auto bound = Bind(CoordinateView<DeviceMemorySpace>(
                      GetCoordinateSystem(), MakeConstRank2View(coords)),
                    owners);
  ScaleByMeasures(
    weights,
    detail::ElementMeasuresAtPoints(*this, *bound, owners,
                                    "OmegaHGeometry::CreateQuadrature"));
  // weights carry the cell measures in the mesh coordinates: make them
  // physical measures
  pcms::detail::ScaleByVolumeFactor(weights, *GetCoordinateSystem(),
                                    MakeConstRank2View(coords));
  return Quadrature{std::move(bound), weights, {}};
}

} // namespace pcms::omega_h
