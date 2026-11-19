#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/geometry/quadrature.hpp>
#include <pcms/backends/omega_h/omega_h.hpp>
#include <pcms/backends/omega_h/omega_h_geometry.hpp>
#include <pcms/transfer/omega_h_intersection_quadrature.hpp>
#include "field_test_utils.hpp"
#include <cmath>
#include <functional>

namespace
{

// sum_p w_p f(x_p) over a quadrature, on the host
double Integrate(const pcms::Quadrature& quadrature,
                 const std::function<double(const double*)>& f)
{
  const auto points = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace{}, quadrature.points->LocatedPointValues());
  const auto weights = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace{}, quadrature.weights);
  REQUIRE(points.extent(0) == weights.extent(0));
  double sum = 0;
  for (std::size_t p = 0; p < points.extent(0); ++p) {
    double x[3] = {points(p, 0), points(p, 1),
                   points.extent(1) > 2 ? points(p, 2) : 0.0};
    sum += weights(p) * f(x);
  }
  return sum;
}

double WeightSum(const pcms::Quadrature& quadrature)
{
  return Integrate(quadrature, [](const double*) { return 1.0; });
}

// The integral of x^a y^b z^c over the unit square or cube
double ExactMonomial(int a, int b, int c, int dim)
{
  const double exact = 1.0 / ((a + 1) * (b + 1));
  return dim == 3 ? exact / (c + 1) : exact;
}

// Quadratures over an Omega_h geometry
template <typename Make>
void CheckPolynomialExactness(Omega_h::Mesh& mesh, Make make)
{
  const int dim = mesh.dim();
  auto geometry = pcms::omega_h::CreateGeometry(
    mesh, pcms::detail::DeferredCartesian());
  const int max_order = GENERATE(1, 2);
  CAPTURE(dim, max_order);
  const auto quadrature = make(*geometry, max_order);
  CHECK(quadrature.points->GetGeometry() == geometry);
  CHECK(quadrature.points->MissingPoints().extent(0) == 0);
  CHECK(WeightSum(quadrature) == Catch::Approx(1.0).margin(1e-12));
  for (int a = 0; a <= max_order; ++a) {
    for (int b = 0; a + b <= max_order; ++b) {
      for (int c = 0; a + b + c <= max_order && (dim == 3 || c == 0); ++c) {
        CAPTURE(a, b, c);
        const double integral = Integrate(quadrature, [&](const double* x) {
          return std::pow(x[0], a) * std::pow(x[1], b) * std::pow(x[2], c);
        });
        CHECK(integral ==
              Catch::Approx(ExactMonomial(a, b, c, dim)).margin(1e-12));
      }
    }
  }
}

} // namespace

TEST_CASE("CreateQuadrature: the reference rule integrates polynomials up to "
          "its order exactly (2D)",
          "[quadrature]")
{
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildSquareMesh(lib, 3);
  CheckPolynomialExactness(mesh, [](const pcms::Geometry& geometry, int order) {
    return geometry.CreateQuadrature(pcms::QuadratureOptions{.order = order});
  });
}

TEST_CASE("CreateQuadrature: the reference rule integrates polynomials up to "
          "its order exactly (3D)",
          "[quadrature]")
{
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildUnitCube(lib, 2);
  CheckPolynomialExactness(mesh, [](const pcms::Geometry& geometry, int order) {
    return geometry.CreateQuadrature(pcms::QuadratureOptions{.order = order});
  });
}

TEST_CASE("CreateQuadrature: Monte Carlo weights sum to the area and the "
          "estimate converges; the seed fixes the points",
          "[quadrature]")
{
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildSquareMesh(lib, 3);
  auto geometry =
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian());
  const auto square = [](const double* x) { return x[0] * x[0]; };
  double previous_error = 1.0;
  for (const int samples : {4, 64, 1024}) {
    const auto quadrature = geometry->CreateQuadrature(
      pcms::QuadratureOptions{.samples_per_cell = samples});
    CAPTURE(samples);
    REQUIRE(quadrature.points->NumPoints() == mesh.nelems() * samples);
    CHECK(quadrature.points->MissingPoints().extent(0) == 0);
    CHECK(WeightSum(quadrature) == Catch::Approx(1.0).margin(1e-12));
    const double error = std::abs(Integrate(quadrature, square) - 1.0 / 3.0);
    CAPTURE(error);
    previous_error = error;
  }
  CHECK(previous_error < 0.01);

  const pcms::QuadratureOptions options{.samples_per_cell = 8, .seed = 7};
  const auto first = geometry->CreateQuadrature(options);
  const auto second = geometry->CreateQuadrature(options);
  CHECK(Integrate(first, square) == Integrate(second, square));
  const auto third = geometry->CreateQuadrature(
    pcms::QuadratureOptions{.samples_per_cell = 8, .seed = 8});
  CHECK(Integrate(first, square) != Integrate(third, square));
}

TEST_CASE("CreateQuadrature: unsupported rules throw", "[quadrature]")
{
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildSquareMesh(lib, 1);
  auto geometry =
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian());
  CHECK_THROWS_AS(
    geometry->CreateQuadrature(pcms::QuadratureOptions{.order = 0}),
    pcms::pcms_error);
  // MeshFields' simplex rules go up to order 2
  CHECK_THROWS_AS(
    geometry->CreateQuadrature(pcms::QuadratureOptions{.order = 3}),
    pcms::pcms_error);
}

TEST_CASE("CreateIntersectionQuadrature: weights sum to the domain, the rule "
          "is exact on the pieces, and both meshes are bound without a miss",
          "[quadrature][intersection]")
{
  Omega_h::Library lib;
  const int dim = GENERATE(2, 3);
  auto source_mesh = dim == 2 ? pcms::test::BuildSquareMesh(lib, 3)
                              : pcms::test::BuildUnitCube(lib, 2);
  auto target_mesh = dim == 2 ? pcms::test::BuildUnitSquare(lib, 0)
                              : pcms::test::BuildUnitCube(lib, 1);
  const auto source = std::dynamic_pointer_cast<const pcms::omega_h::OmegaHGeometry>(
    pcms::omega_h::CreateGeometry(source_mesh,
                                  pcms::detail::DeferredCartesian()));
  const auto target = std::dynamic_pointer_cast<const pcms::omega_h::OmegaHGeometry>(
    pcms::omega_h::CreateGeometry(target_mesh,
                                  pcms::detail::DeferredCartesian()));
  const int order = 2;
  const auto quadrature = pcms::omega_h::CreateIntersectionQuadrature(
    source, target, pcms::QuadratureOptions{.order = order});
  CHECK(quadrature.points->GetGeometry() == target);
  CHECK(quadrature.points->MissingPoints().extent(0) == 0);
  CHECK(quadrature.other.geometry == source);
  CHECK(quadrature.other.entities.extent(0) ==
        static_cast<std::size_t>(quadrature.points->NumPoints()));
  CHECK(WeightSum(quadrature) == Catch::Approx(1.0).margin(1e-12));
  for (int a = 0; a <= order; ++a) {
    for (int b = 0; a + b <= order; ++b) {
      const int c = dim == 3 ? std::max(0, order - a - b) : 0;
      const double integral = Integrate(quadrature, [&](const double* x) {
        return std::pow(x[0], a) * std::pow(x[1], b) * std::pow(x[2], c);
      });
      CAPTURE(a, b, c);
      CHECK(integral == Catch::Approx(ExactMonomial(a, b, c, dim)).margin(1e-12));
    }
  }
  // the source binds to the points in the elements they were generated in
  const pcms::CoordinateView<pcms::DeviceMemorySpace> points(
    quadrature.points->GetCoordinateSystem(),
    pcms::MakeConstRank2View(quadrature.points->LocatedPointValues()));
  const auto bound = source->Bind(points, quadrature.other);
  CHECK(bound->MissingPoints().extent(0) == 0);
}
