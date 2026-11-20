#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include "embedding_test_utils.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pcms::DeviceMemorySpace;
using pcms::Real;
using pcms::test::Annulus;

namespace
{

// the analytic Jacobian J(i, j) = d x_j / d xi_i of the annulus
std::array<std::array<Real, 2>, 2> AnnulusJacobian(Real xi1, Real xi2)
{
  const Real rho = Annulus::Rho(xi1);
  const Real theta = Annulus::Theta(xi2);
  return {{{0.5 * std::cos(theta), 0.5 * std::sin(theta)},
           {-2.0 * rho * std::sin(theta), 2.0 * rho * std::cos(theta)}}};
}

const std::vector<Real> Params = {0.31, 0.42, 0.77, 0.13, 0.5, 0.95, 0.05, 0.6};
// the position is a spline of 32 x 32 cells: a smooth map is reproduced to
// roughly its fourth derivative times h^4
constexpr Real SplineTol = 2e-6;

struct Fixture
{
  std::shared_ptr<const pcms::CoordinateMap> embedding = pcms::CreateEmbedding(
    pcms::test::MakeSplinePosition(pcms::test::MakeUnitSquareGrid(32),
                                   Annulus::Position),
    pcms::csys::CreateCylindricalRZ());
  pcms::test::DeviceCoordinates xi = pcms::test::CreateDeviceCoordinateView(
    Params, embedding->GetSourceCoordinateSystem(), 2);
};

} // namespace

TEST_CASE("An embedding maps parameter points and gives its Jacobian",
          "[embedding]")
{
  const Fixture f;
  CHECK(f.embedding->GetTargetCoordinateSystem()->Kind() == "cylindrical_rz");
  CHECK(f.embedding->GetSourceCoordinateSystem()->Dimension() == 2);
  const auto bound = f.embedding->Bind(f.xi.coordinate_view);
  const auto mapped = pcms::test::CopyCoordinatesToHost(bound->Mapped());
  const auto jacobian_view = bound->Jacobian();
  Kokkos::View<Real***, pcms::HostMemorySpace> jacobian("j", 4, 2, 2);
  Kokkos::View<Real***, DeviceMemorySpace> device("j", 4, 2, 2);
  Kokkos::parallel_for(
    "copy", Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, 4),
    KOKKOS_LAMBDA(const pcms::LO p) {
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          device(p, i, j) = jacobian_view(p, i, j);
        }
      }
    });
  Kokkos::deep_copy(jacobian, device);
  for (int p = 0; p < 4; ++p) {
    const Real xi1 = Params[std::size_t{2} * p];
    const Real xi2 = Params[2 * p + 1];
    const auto x = Annulus::Position(xi1, xi2);
    const auto j = AnnulusJacobian(xi1, xi2);
    CAPTURE(p);
    CHECK_THAT(mapped(p, 0), WithinAbs(x[0], SplineTol));
    CHECK_THAT(mapped(p, 1), WithinAbs(x[1], SplineTol));
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        CHECK_THAT(jacobian(p, a, b), WithinAbs(j[a][b], 100 * SplineTol));
      }
    }
  }
}

TEST_CASE("The parameter system has the induced metric", "[embedding]")
{
  const Fixture f;
  const auto system = f.embedding->GetSourceCoordinateSystem();
  CHECK_FALSE(system->IsConstantMetric());
  const auto bound = system->Bind(f.xi.coordinate_view.GetValues());
  const auto metric = bound->Metric();
  const auto volume = bound->VolumeFactor();
  Kokkos::View<Real***, DeviceMemorySpace> g("g", 4, 2, 2);
  Kokkos::View<Real*, DeviceMemorySpace> v("v", 4);
  Kokkos::parallel_for(
    "copy", Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, 4),
    KOKKOS_LAMBDA(const pcms::LO p) {
      v(p) = volume(p);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          g(p, i, j) = metric(p, i, j);
        }
      }
    });
  const auto g_host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), g);
  const auto v_host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), v);
  for (int p = 0; p < 4; ++p) {
    const Real rho = Annulus::Rho(Params[std::size_t{2} * p]);
    const auto x = Annulus::Position(Params[std::size_t{2} * p], Params[std::size_t{2} * p + 1]);
    CAPTURE(p);
    CHECK_THAT(g_host(p, 0, 0), WithinAbs(0.25, 100 * SplineTol));
    CHECK_THAT(g_host(p, 1, 1), WithinAbs(4 * rho * rho, 100 * SplineTol));
    CHECK_THAT(g_host(p, 0, 1), WithinAbs(0.0, 100 * SplineTol));
    // sqrt(det g) = rho, times the volume factor of (R, Z), R
    CHECK_THAT(v_host(p), WithinAbs(rho * x[0], 100 * SplineTol));
  }
}

TEST_CASE("The bound embedding re-expresses vectors and covectors",
          "[embedding]")
{
  const Fixture f;
  const auto system = f.embedding->GetSourceCoordinateSystem();
  const auto bound = f.embedding->Bind(f.xi.coordinate_view);
  for (const bool covariant : {false, true}) {
    Kokkos::View<Real**, DeviceMemorySpace> data("values", 4, 2);
    const auto host = Kokkos::create_mirror_view(data);
    for (int p = 0; p < 4; ++p) {
      host(p, 0) = 1.0;
      host(p, 1) = 0.0;
    }
    Kokkos::deep_copy(data, host);
    const pcms::ValueBasis stored{system, pcms::ComponentScaling::Natural,
                                  covariant ? pcms::values::Covector
                                            : pcms::values::Vector};
    const auto out =
      bound->TransformValues(pcms::ValueView<Real, DeviceMemorySpace>(
        stored, pcms::MakeRank2View(data)));
    CHECK(out.GetBasis().system->Kind() == "cylindrical_rz");
    CHECK(out.GetBasis().scaling == pcms::ComponentScaling::Physical);
    const auto result =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), data);
    for (int p = 0; p < 4; ++p) {
      const Real theta = Annulus::Theta(Params[2 * p + 1]);
      // the vector e_xi1 is d x / d xi1 = 0.5 (cos, sin); the covector
      // dxi1 has the physical components 2 (cos, sin)
      const Real scale = covariant ? 2.0 : 0.5;
      CAPTURE(p, covariant);
      CHECK_THAT(result(p, 0),
                 WithinAbs(scale * std::cos(theta), 100 * SplineTol));
      CHECK_THAT(result(p, 1),
                 WithinAbs(scale * std::sin(theta), 100 * SplineTol));
    }
  }
}

TEST_CASE("Embeddings refuse what they cannot do", "[embedding]")
{
  const auto geometry = pcms::test::MakeUnitSquareGrid(4);
  const auto position =
    pcms::test::MakeSplinePosition(geometry, Annulus::Position);
  CHECK_THROWS_WITH(
    pcms::CreateEmbedding(nullptr, pcms::csys::CreateCylindricalRZ()),
    ContainsSubstring("must not be null"));
  CHECK_THROWS_WITH(
    pcms::CreateEmbedding(position, pcms::csys::CreateCartesian(3)),
    ContainsSubstring("same dimension"));
  const Fixture f;
  const auto bound = f.embedding->Bind(f.xi.coordinate_view);
  Kokkos::View<Real**, DeviceMemorySpace> data("values", 4, 2);
  CHECK_THROWS_WITH(
    bound->TransformValues(pcms::ValueView<Real, DeviceMemorySpace>(
      pcms::ValueBasis{pcms::csys::CreateCylindricalRZ(),
                       pcms::ComponentScaling::Physical, pcms::values::Vector},
      pcms::MakeRank2View(data))),
    ContainsSubstring("natural components"));
}
