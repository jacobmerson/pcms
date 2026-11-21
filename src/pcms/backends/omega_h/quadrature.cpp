#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/backends/omega_h/simplex_quadrature.hpp"
#include "pcms/configuration.hpp"
#include "pcms/geometry/detail/geometry.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/assert.hpp"
#include <Kokkos_Random.hpp>
#include <Omega_h_shape.hpp>

namespace pcms::omega_h
{

namespace
{

using Pool = Kokkos::Random_XorShift64_Pool<DefaultExecutionSpace>;

// Normalized exponential draws are uniform on the simplex (Dirichlet(1, ...)).
template <int Dim>
KOKKOS_INLINE_FUNCTION Omega_h::Vector<Dim + 1> UniformBarycentric(
  Pool::generator_type& gen)
{
  Omega_h::Vector<Dim + 1> b;
  Real sum = 0;
  for (int k = 0; k <= Dim; ++k) {
    b[k] = -Kokkos::log(1.0 - gen.drand());
    sum += b[k];
  }
  for (int k = 0; k <= Dim; ++k) {
    b[k] /= sum;
  }
  return b;
}

template <int Dim>
Quadrature CreateSimplexQuadrature(const OmegaHGeometry& geometry,
                                   const QuadratureOptions& options)
{
  constexpr Real ReferenceMeasure = Dim == 3 ? 1.0 / 6.0 : 0.5;
  const bool random = options.samples_per_cell > 0;
  int per_cell = options.samples_per_cell;
  Kokkos::View<Real * [Dim + 1], DeviceMemorySpace> bary;
  Kokkos::View<Real*, DeviceMemorySpace> rule_weights;
  Pool pool;
  if (random) {
    pool.init(options.seed, DefaultExecutionSpace().concurrency());
  } else {
#ifdef PCMS_ENABLE_OMEGA_H_MESHFIELDS
    const detail::IntegrationData<Dim> rule(options.order);
    bary = rule.bary_coords;
    rule_weights = rule.weights;
    per_cell = rule.size();
#else
    throw pcms_error("OmegaHGeometry::CreateQuadrature: the reference rules "
                     "need PCMS_ENABLE_OMEGA_H_MESHFIELDS");
#endif
  }
  Omega_h::Mesh& mesh = geometry.GetMesh();
  const LO nelems = mesh.nelems();
  const auto num_points = static_cast<std::size_t>(nelems) * per_cell;
  Kokkos::View<LO*, DeviceMemorySpace> elements("quadrature_elements",
                                                num_points);
  Kokkos::View<Real**, DeviceMemorySpace> coords("quadrature_coords",
                                                 num_points, Dim);
  Kokkos::View<Real*, DeviceMemorySpace> weights("quadrature_weights",
                                                 num_points);
  const auto elem_verts = mesh.ask_elem_verts();
  const auto x = mesh.coords();
  const auto determinant = geometry.CellJacobian().determinant;
  Kokkos::parallel_for(
    "OmegaHGeometry::CreateQuadrature",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, nelems),
    KOKKOS_LAMBDA(const LO e) {
      const auto verts = Omega_h::gather_verts<Dim + 1>(elem_verts, e);
      const Real measure = Kokkos::fabs(determinant(e));
      const auto fill = [&](int q, const Omega_h::Vector<Dim + 1>& b, Real w) {
        const LO i = e * per_cell + q;
        elements(i) = e;
        weights(i) = w * measure;
        for (int d = 0; d < Dim; ++d) {
          coords(i, d) = 0;
          for (int k = 0; k <= Dim; ++k) {
            coords(i, d) += b[k] * x[verts[k] * Dim + d];
          }
        }
      };
      if (random) {
        auto gen = pool.get_state();
        for (int q = 0; q < per_cell; ++q) {
          fill(q, UniformBarycentric<Dim>(gen), ReferenceMeasure / per_cell);
        }
        pool.free_state(gen);
      } else {
        for (int q = 0; q < per_cell; ++q) {
          Omega_h::Vector<Dim + 1> b;
          for (int k = 0; k <= Dim; ++k) {
            b[k] = bary(q, k);
          }
          fill(q, b, rule_weights(q));
        }
      }
    });
  auto bound =
    geometry.Bind(CoordinateView<DeviceMemorySpace>(
                    geometry.GetCoordinateSystem(), MakeConstRank2View(coords)),
                  OwnerHints{geometry.shared_from_this(), Dim, elements});
  pcms::detail::ScaleByVolumeFactor(weights, *bound);
  return Quadrature{std::move(bound), weights, {}};
}

} // namespace

Quadrature OmegaHGeometry::CreateQuadrature(
  const QuadratureOptions& options) const
{
  if (options.samples_per_cell <= 0 && options.order < 1) {
    throw pcms_error("OmegaHGeometry::CreateQuadrature: the order must be "
                     "positive");
  }
  return mesh_.dim() == 3 ? CreateSimplexQuadrature<3>(*this, options)
                          : CreateSimplexQuadrature<2>(*this, options);
}

} // namespace pcms::omega_h
