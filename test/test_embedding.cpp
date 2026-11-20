#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include "embedding_test_utils.hpp"
#include <numbers>

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
    const Real xi1 = Params[2 * p];
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
    const Real rho = Annulus::Rho(Params[2 * p]);
    const auto x = Annulus::Position(Params[2 * p], Params[2 * p + 1]);
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

namespace
{

// An embedded annulus sector, and a P1 space on its parameter grid put on it.
struct EmbeddedFixture
{
  std::shared_ptr<const pcms::Geometry> grid =
    pcms::test::MakeUnitSquareGrid(32);
  std::shared_ptr<const pcms::CoordinateMap> embedding = pcms::CreateEmbedding(
    pcms::test::MakeSplinePosition(grid, Annulus::Position),
    pcms::csys::CreateCylindricalRZ());
  std::shared_ptr<const pcms::Geometry> embedded =
    pcms::CreateEmbeddedGeometry(grid, embedding);
  std::shared_ptr<const pcms::FunctionSpace> space =
    pcms::CreateEmbeddedSpace(pcms::CreateLagrangeSpace(grid, 1, 1), embedded);
};

// f = xi1 + 2 xi2 at the points of `bound`
std::vector<Real> EvaluateLinearInParameters(
  const EmbeddedFixture& f,
  const std::shared_ptr<const pcms::BoundPoints>& bound)
{
  auto field = f.space->CreateFunction<Real>("f");
  pcms::test::SetField(
    field, KOKKOS_LAMBDA(Real xi1, Real xi2) { return xi1 + 2 * xi2; });
  const auto evaluator = f.space->CreatePointEvaluator<Real>(bound);
  Kokkos::View<Real**, DeviceMemorySpace> out("out", bound->NumPoints(), 1);
  evaluator->Evaluate(field, pcms::test::TagLike(field, out));
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
  return std::vector<Real>(host.data(), host.data() + host.extent(0));
}

} // namespace

TEST_CASE("An embedded geometry locates physical points by inverting the "
          "embedding",
          "[embedding][embedded_geometry]")
{
  const EmbeddedFixture f;
  CHECK(f.embedded->GetCoordinateSystem()->Kind() == "cylindrical_rz");
  CHECK(f.embedded->TopologicalDimension() == 2);
  CHECK(*f.embedded->GetParameterSystem() ==
        *f.embedding->GetSourceCoordinateSystem());
  CHECK(f.embedded->GetEmbedding() == f.embedding);

  std::vector<Real> physical;
  std::vector<Real> expected;
  for (int i = 0; i < 4; ++i) {
    const auto x = Annulus::Position(Params[2 * i], Params[2 * i + 1]);
    physical.insert(physical.end(), x.begin(), x.end());
    expected.push_back(Params[2 * i] + 2 * Params[2 * i + 1]);
  }
  const auto bound = pcms::test::Bind(*f.embedded, physical);
  CHECK(bound->NumPoints() == 4);
  // the parameters are recovered to the error of the position's spline
  const auto values = EvaluateLinearInParameters(f, bound);
  for (int i = 0; i < 4; ++i) {
    CAPTURE(i);
    CHECK_THAT(values[i], WithinAbs(expected[i], 1e-5));
  }

  // the Jacobian at the found points: |det J| = rho
  const auto differential = bound->Differential();
  REQUIRE(differential.jacobian.extent(0) == 4);
  REQUIRE_FALSE(differential.per_element);
  Kokkos::View<Real*, DeviceMemorySpace> det("det", 4);
  const auto determinant = differential.determinant;
  Kokkos::parallel_for(
    "copy", Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, 4),
    KOKKOS_LAMBDA(const pcms::LO p) { det(p) = determinant(p); });
  const auto det_host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), det);
  for (int i = 0; i < 4; ++i) {
    CHECK_THAT(det_host(i), WithinAbs(Annulus::Rho(Params[2 * i]), 1e-4));
  }
}

TEST_CASE("An embedded geometry binds points of its parameter system without "
          "a search",
          "[embedding][embedded_geometry]")
{
  const EmbeddedFixture f;
  const auto coords = pcms::test::CreateDeviceCoordinateView(
    Params, f.embedded->GetParameterSystem(), 2);
  const auto bound = f.embedded->Bind(coords.coordinate_view);
  // the field is evaluated at exactly the parameters given
  const auto values = EvaluateLinearInParameters(f, bound);
  for (int i = 0; i < 4; ++i) {
    CHECK_THAT(values[i], WithinAbs(Params[2 * i] + 2 * Params[2 * i + 1],
                                    pcms::test::ExactTol));
  }
  // outside the parameter grid
  const auto outside = pcms::test::CreateDeviceCoordinateView(
    {1.5, 0.5}, f.embedded->GetParameterSystem(), 2);
  CHECK_THROWS(f.embedded->Bind(outside.coordinate_view));
  const auto missing = f.embedded->Bind(
    outside.coordinate_view,
    pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, -7.0});
  CHECK(missing->MissingPoints().extent(0) == 1);
  CHECK(EvaluateLinearInParameters(f, missing)[0] == -7.0);
}

TEST_CASE("An embedded geometry applies the out-of-bounds policy to points "
          "outside its image",
          "[embedding][embedded_geometry]")
{
  const EmbeddedFixture f;
  // inside the sector, the hole of the annulus, and outside
  const std::vector<Real> points = {
    Annulus::Rc + 0.75, 0.3, Annulus::Rc, 0.0, 5.0, 5.0};
  CHECK_THROWS_AS(pcms::test::Bind(*f.embedded, points), pcms::pcms_error);
  CHECK_THROWS_AS(pcms::test::Bind(*f.embedded, points,
                                   pcms::OutOfBoundsPolicy{
                                     pcms::OutOfBoundsMode::NEAREST_BOUNDARY}),
                  pcms::pcms_error);
  const auto bound = pcms::test::Bind(
    *f.embedded, points,
    pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, -7.0});
  CHECK(bound->FoundPoints().extent(0) == 1);
  CHECK(bound->MissingPoints().extent(0) == 2);
  const auto values = EvaluateLinearInParameters(f, bound);
  CHECK(values[1] == -7.0);
  CHECK(values[2] == -7.0);
}

TEST_CASE("The quadrature of an embedded geometry integrates in the physical "
          "measure",
          "[embedding][embedded_geometry][quadrature]")
{
  const EmbeddedFixture f;
  const auto quadrature = f.embedded->CreateQuadrature({.order = 4});
  CHECK(quadrature.points->NumPoints() == 32 * 32 * 9);
  const auto weights = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), quadrature.weights);
  Real sum = 0;
  for (size_t i = 0; i < weights.extent(0); ++i) {
    sum += weights(i);
  }
  // the area of the sector times the R of the volume of revolution
  CHECK_THAT(sum, WithinAbs(Annulus::MomentOfR(), 1e-5));
  CHECK_THROWS(f.embedded->CreateQuadrature({.samples_per_cell = 3}));
  // the points are bound on the embedded geometry
  CHECK(quadrature.points->GetGeometry() == f.embedded);
}

TEST_CASE("An embedded geometry survives a singular point of the embedding",
          "[embedding][embedded_geometry]")
{
  // a disc of radius 1 around (2, 0) with the center a pole: rho = xi1,
  // theta = 2 pi xi2, J has the determinant 2 pi rho, zero on the edge xi1 = 0
  const auto position = [](Real xi1, Real xi2) {
    const Real theta = 2 * std::numbers::pi * xi2;
    return std::array<Real, 2>{2.0 + xi1 * std::cos(theta),
                               xi1 * std::sin(theta)};
  };
  const auto grid = pcms::test::MakeUnitSquareGrid(32);
  const auto embedding =
    pcms::CreateEmbedding(pcms::test::MakeSplinePosition(grid, position),
                          pcms::csys::CreateCylindricalRZ());
  const auto embedded = pcms::CreateEmbeddedGeometry(grid, embedding);
  // the pole and a regular point; the policy fills what is not found
  const auto bound = pcms::test::Bind(
    *embedded, {2.0, 0.0, 2.0, 0.5},
    pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, -7.0});
  CHECK(bound->NumPoints() == 2);
  // the regular point, xi = (0.5, 0.25), is found
  const auto found = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), bound->FoundPoints());
  bool regular_found = false;
  for (size_t i = 0; i < found.extent(0); ++i) {
    regular_found = regular_found || found(i) == 1;
  }
  CHECK(regular_found);
}

TEST_CASE("Embedded geometries refuse what they cannot do",
          "[embedding][embedded_geometry]")
{
  const EmbeddedFixture f;
  CHECK_THROWS_WITH(pcms::CreateEmbeddedGeometry(nullptr, f.embedding),
                    ContainsSubstring("CreateEmbedding of a function"));
  // an embedding of another geometry
  CHECK_THROWS_WITH(pcms::CreateEmbeddedGeometry(
                      pcms::test::MakeUnitSquareGrid(4), f.embedding),
                    ContainsSubstring("CreateEmbedding of a function"));
  CHECK_THROWS_WITH(
    pcms::CreateEmbeddedSpace(
      pcms::CreateLagrangeSpace(pcms::test::MakeUnitSquareGrid(4), 1, 1),
      f.embedded),
    ContainsSubstring("embedded geometry of the space"));
  const auto bound = pcms::test::Bind(*f.embedded, {Annulus::Rc + 0.75, 0.3});
  CHECK_FALSE(f.space->Supports(pcms::DifferentialOperator::Gradient));
  // Rebind locates new points
  const auto again = f.embedded->Rebind(
    bound, pcms::test::CreateDeviceCoordinateView(
             {Annulus::Rc + 0.8, 0.2}, f.embedded->GetCoordinateSystem(), 2)
             .coordinate_view);
  CHECK(again->NumPoints() == 1);
}
