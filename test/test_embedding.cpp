#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include "embedding_test_utils.hpp"
#include <random>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pcms::DeviceMemorySpace;
using pcms::Real;
using pcms::test::Annulus;

namespace
{

// the analytic Jacobian J(j, i) = d x_j / d xi_i of the annulus
std::array<std::array<Real, 2>, 2> AnnulusJacobian(Real xi1, Real xi2)
{
  const Real rho = Annulus::Rho(xi1);
  const Real theta = Annulus::Theta(xi2);
  return {{{0.5 * std::cos(theta), -2.0 * rho * std::sin(theta)},
           {0.5 * std::sin(theta), 2.0 * rho * std::cos(theta)}}};
}

const std::vector<Real> Params = {0.31, 0.42, 0.77, 0.13, 0.5, 0.95, 0.05, 0.6};
// the position is a spline of 32 x 32 cells: a smooth map is reproduced to
// roughly its fourth derivative times h^4
constexpr Real SplineTol = 2e-6;

struct Fixture
{
  std::shared_ptr<const pcms::Function<Real>> position =
    pcms::test::MakeSplinePosition(pcms::test::MakeUnitSquareGrid(32),
                                   Annulus::Position);
  std::shared_ptr<const pcms::CoordinateMap> map =
    pcms::CreateCoordinateMap(position);
  pcms::test::DeviceCoordinates xi = pcms::test::CreateDeviceCoordinateView(
    Params, map->GetSourceCoordinateSystem(), 2);
};

} // namespace

TEST_CASE("The coordinate map of a position maps parameter points and gives "
          "its Jacobian",
          "[embedding]")
{
  const Fixture f;
  CHECK(f.map->GetTargetCoordinateSystem()->Kind() == "cylindrical_rz");
  CHECK(f.map->GetSourceCoordinateSystem()->Kind() == "xi");
  const auto bound = f.map->Bind(f.xi.coordinate_view);
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

TEST_CASE("Coordinates pass through value transforms", "[embedding]")
{
  const Fixture f;
  const auto bound = f.map->Bind(f.xi.coordinate_view);
  const auto coordinates =
    pcms::ValueBasis::Coordinates(pcms::csys::CreateCartesian(3));
  CHECK(bound->ValueTransforms(coordinates).empty());
  const pcms::ValueBasis vector{pcms::csys::CreateCylindricalRZ(),
                                pcms::ComponentScaling::Physical,
                                pcms::values::Vector};
  for (const auto* transform : bound->ValueTransforms(vector)) {
    CHECK(transform->OutputBasis(coordinates) == coordinates);
  }
  // coordinates are not scalars, and coordinates of different coordinate
  // systems differ
  CHECK_FALSE(pcms::detail::SameValueBasis(coordinates, pcms::ValueBasis{}));
  CHECK_FALSE(pcms::detail::SameValueBasis(
    coordinates,
    pcms::ValueBasis::Coordinates(pcms::csys::CreateCylindricalRThetaZ())));
  CHECK(pcms::detail::SameValueBasis(
    f.position->GetData().GetValueBasis(),
    pcms::ValueBasis::Coordinates(pcms::csys::CreateCylindricalRZ())));
}

TEST_CASE("A function of several components defaults to independent "
          "components",
          "[embedding]")
{
  const auto space =
    pcms::CreateLagrangeSpace(pcms::test::MakeUnitSquareGrid(2), 1, 2);
  CHECK(space->CreateFunction<Real>().GetData().GetValueBasis().kind ==
        pcms::ValueKind::Array);
  CHECK_THROWS_WITH(space->CreateFunction<Real>("x", pcms::ValueBasis{}),
                    ContainsSubstring("a scalar has one component"));
}

TEST_CASE("The coordinate map of a position refuses what it cannot do",
          "[embedding]")
{
  const auto geometry = pcms::test::MakeUnitSquareGrid(4);
  CHECK_THROWS_WITH(pcms::CreateCoordinateMap(nullptr),
                    ContainsSubstring("must not be null"));
  // values that are not coordinates
  const auto space = pcms::CreateSplineSpace(geometry, 2);
  const auto scalars = std::make_shared<pcms::Function<Real>>(
    space->CreateFunction<Real>("x", {.kind = pcms::ValueKind::Array}));
  CHECK_THROWS_WITH(pcms::CreateCoordinateMap(scalars),
                    ContainsSubstring("must be coordinates"));
  // two components are not the coordinates of a 3D coordinate system
  CHECK_THROWS_WITH(
    space->CreateFunction<Real>(
      "x", pcms::ValueBasis::Coordinates(pcms::csys::CreateCartesian(3))),
    ContainsSubstring("require 3 components"));
}

TEST_CASE("Coordinates on an identified axis interpolate through the nearest "
          "image",
          "[embedding]")
{
  // u = (x, y + x / 2) on a grid periodic in y, with u2 stored modulo 1
  const std::vector<pcms::PeriodicAxis> periodic = {{1, 0.0, 1.0}};
  const auto grid = pcms::test::MakeUnitSquareGrid(
    8, pcms::csys::CreateParameterSystem("xi", 2, periodic));
  const auto position = pcms::test::MakePositionFunction(
    pcms::CreateLagrangeSpace(grid, 1, 2),
    [](const Real* xi, Real* u) {
      u[0] = xi[0];
      u[1] = std::fmod(xi[1] + 0.5 * xi[0], 1.0);
    },
    pcms::csys::CreateParameterSystem("u", 2, periodic));
  // across the last cell in y, and inside cells where u2 wraps
  const std::vector<Real> points = {0.3, 0.95, 0.6, 0.5, 0.95, 0.99};
  const auto xi = pcms::test::CreateDeviceCoordinateView(
    points, grid->GetCoordinateSystem(), 2);
  const auto bound =
    pcms::CreateCoordinateMap(position)->Bind(xi.coordinate_view);
  const auto mapped = pcms::test::CopyCoordinatesToHost(bound->Mapped());
  const auto j = bound->Jacobian();
  Kokkos::View<Real**, DeviceMemorySpace> jacobian("j", 3, 4);
  Kokkos::parallel_for(
    "copy", Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, 3),
    KOKKOS_LAMBDA(const pcms::LO p) {
      for (int k = 0; k < 4; ++k) {
        jacobian(p, k) = j(p, k / 2, k % 2);
      }
    });
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), jacobian);
  for (int p = 0; p < 3; ++p) {
    CAPTURE(p);
    const Real x = points[std::size_t{2} * p];
    const Real y = points[std::size_t{2} * p + 1];
    CHECK_THAT(mapped(p, 0), WithinAbs(x, 1e-12));
    CHECK_THAT(std::remainder(mapped(p, 1) - y - 0.5 * x, 1.0),
               WithinAbs(0.0, 1e-12));
    const std::array<Real, 4> expected = {1.0, 0.0, 0.5, 1.0};
    for (int k = 0; k < 4; ++k) {
      CHECK_THAT(host(p, k), WithinAbs(expected[k], 1e-12));
    }
  }
  // the evaluation through the nearest image has no transpose
  const auto& space = position->GetSpace();
  const auto evaluator =
    space.CreatePointEvaluator<Real>(grid->Bind(xi.coordinate_view));
  const auto& basis = position->GetData().GetValueBasis();
  auto functional = space.CreateFunction<Real>("functional", basis);
  const Kokkos::View<Real**, DeviceMemorySpace> loads("loads", 3, 2);
  CHECK_THROWS_WITH(
    evaluator->EvaluateTranspose(
      pcms::ValueView<const Real, DeviceMemorySpace>(
        evaluator->OutputBasis(basis), pcms::MakeConstRank2View(loads)),
      functional),
    ContainsSubstring("nearest image"));
}

namespace
{

struct MappedPoints
{
  Kokkos::View<const Real**, pcms::HostMemorySpace> x;
  // jacobian(p, a * dim + b) = d x_a / d xi_b
  Kokkos::View<Real**, pcms::HostMemorySpace> jacobian;
};

// The image of `points` of `grid` under `position`, and its Jacobian
MappedPoints MapPoints(
  const std::shared_ptr<const pcms::Function<Real>>& position,
  const std::shared_ptr<const pcms::Geometry>& grid,
  const std::vector<Real>& points, int dim)
{
  const auto xi = pcms::test::CreateDeviceCoordinateView(
    points, grid->GetCoordinateSystem(), dim);
  const auto bound =
    pcms::CreateCoordinateMap(position)->Bind(xi.coordinate_view);
  const auto j = bound->Jacobian();
  const auto n = static_cast<pcms::LO>(j.extent(0));
  Kokkos::View<Real**, Kokkos::LayoutRight, DeviceMemorySpace> jacobian(
    "j", n, static_cast<std::size_t>(dim) * dim);
  Kokkos::parallel_for(
    "copy", Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const pcms::LO p) {
      for (int k = 0; k < dim * dim; ++k) {
        jacobian(p, k) = j(p, k / dim, k % dim);
      }
    });
  return {
    pcms::test::CopyCoordinatesToHost(bound->Mapped()),
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), jacobian)};
}

// `count` random points with coordinate d in [lower[d], 1)
std::vector<Real> RandomPoints(int count, const std::vector<Real>& lower)
{
  std::mt19937 rng(7);
  std::vector<Real> points;
  for (int p = 0; p < count; ++p) {
    for (const Real l : lower) {
      points.push_back(std::uniform_real_distribution<Real>(l, 1.0)(rng));
    }
  }
  return points;
}

} // namespace

TEST_CASE("Coordinates across a sheared seam interpolate through the image "
          "nearest their neighbour along the seam axis",
          "[embedding]")
{
  // phi = P z + zShift(x, y), z wrapped; zShift jumps 2 P / 5 between y nodes
  constexpr int N = 6;
  constexpr Real P = 2 * std::numbers::pi / 5;
  pcms::UniformGrid<3> cells;
  cells.bot_left = {0.0, 0.0, 0.0};
  cells.edge_length = {1.0, 1.0, 1.0};
  cells.divisions = {N, N, N};
  const auto grid = pcms::uniform_grid::CreateGeometry(
    cells, pcms::csys::CreateParameterSystem("xi", 3, {{2, 0.0, 1.0}}));
  const auto position = pcms::test::MakePositionFunction(
    pcms::CreateLagrangeSpace(grid, 1, 3),
    [](const Real* xi, Real* u) {
      u[0] = xi[0];
      u[1] = xi[1];
      u[2] = P * xi[2] + 0.3 * xi[0] + 0.4 * P * N * xi[1];
    },
    pcms::csys::CreateParameterSystem("u", 3, {{2, 0.0, P}}));
  auto points = RandomPoints(20, {0.0, 0.0, 0.0});
  const auto seam = RandomPoints(20, {0.0, 0.0, (N - 1.0) / N});
  points.insert(points.end(), seam.begin(), seam.end());
  const auto mapped = MapPoints(position, grid, points, 3);
  for (std::size_t p = 0; p < points.size() / 3; ++p) {
    CAPTURE(p);
    const Real* xi = &points[3 * p];
    CHECK_THAT(mapped.x(p, 2),
               WithinAbs(P * xi[2] + 0.3 * xi[0] + 0.4 * P * N * xi[1], 1e-9));
    CHECK_THAT(mapped.jacobian(p, 6), WithinAbs(0.3, 1e-9));
    CHECK_THAT(mapped.jacobian(p, 7), WithinAbs(0.4 * P * N, 1e-9));
    CHECK_THAT(mapped.jacobian(p, 8), WithinAbs(P, 1e-9));
  }
}

TEST_CASE("Longitudes across the seam of a periodic grid interpolate the "
          "unwrapped values",
          "[embedding]")
{
  // longitude = 360 i / N + 40 sin(j) in degrees, i wrapped
  constexpr int N = 8;
  const auto grid = pcms::test::MakeUnitSquareGrid(
    N, pcms::csys::CreateParameterSystem("xi", 2, {{0, 0.0, 1.0}}));
  const auto longitude = [](Real i, Real j) {
    return 360.0 * i / N + 40 * std::sin(j);
  };
  const auto position = pcms::test::MakePositionFunction(
    pcms::CreateLagrangeSpace(grid, 1, 2),
    [&](const Real* xi, Real* u) {
      u[0] = longitude(N * xi[0], N * xi[1]);
      u[1] = 90 * xi[1];
    },
    pcms::csys::CreateParameterSystem("lonlat", 2, {{0, 0.0, 360.0}}));
  const auto points = RandomPoints(20, {(N - 1.0) / N, 0.0});
  const auto mapped = MapPoints(position, grid, points, 2);
  for (std::size_t p = 0; p < points.size() / 2; ++p) {
    CAPTURE(p);
    const Real tx = N * points[2 * p] - (N - 1);
    const Real y = N * points[2 * p + 1];
    const Real j = std::floor(y);
    const Real ty = y - j;
    const Real west =
      (1 - ty) * longitude(N - 1, j) + ty * longitude(N - 1, j + 1);
    const Real east = (1 - ty) * longitude(N, j) + ty * longitude(N, j + 1);
    CHECK_THAT(mapped.x(p, 0), WithinAbs((1 - tx) * west + tx * east, 1e-9));
    CHECK_THAT(mapped.jacobian(p, 0), WithinAbs(N * (east - west), 1e-9));
  }
}

TEST_CASE("Coordinates at a node wrapped along two axes interpolate through "
          "the image nearest each neighbour",
          "[embedding]")
{
  // u = (2 x + 2 y, x - y) on a grid periodic in x and y, both of period 1
  constexpr int N = 6;
  const std::vector<pcms::PeriodicAxis> periodic = {{0, 0.0, 1.0},
                                                    {1, 0.0, 1.0}};
  const auto grid = pcms::test::MakeUnitSquareGrid(
    N, pcms::csys::CreateParameterSystem("xi", 2, periodic));
  const auto position = pcms::test::MakePositionFunction(
    pcms::CreateLagrangeSpace(grid, 1, 2),
    [](const Real* xi, Real* u) {
      u[0] = 2 * xi[0] + 2 * xi[1];
      u[1] = xi[0] - xi[1];
    },
    pcms::csys::CreateParameterSystem("u", 2, periodic));
  const auto points = RandomPoints(20, {(N - 1.0) / N, (N - 1.0) / N});
  const auto mapped = MapPoints(position, grid, points, 2);
  for (std::size_t p = 0; p < points.size() / 2; ++p) {
    CAPTURE(p);
    const Real x = points[2 * p];
    const Real y = points[2 * p + 1];
    CHECK_THAT(mapped.x(p, 0), WithinAbs(2 * x + 2 * y, 1e-9));
    CHECK_THAT(mapped.x(p, 1), WithinAbs(x - y, 1e-9));
    const std::array<Real, 4> expected = {2.0, 2.0, 1.0, -1.0};
    for (int k = 0; k < 4; ++k) {
      CHECK_THAT(mapped.jacobian(p, k), WithinAbs(expected[k], 1e-9));
    }
  }
}

TEST_CASE("Coordinates that jump across the seam inside an open grid "
          "interpolate through the nearest image",
          "[embedding]")
{
  // theta = 2 pi (0.7 + 0.6 xi1 + 0.1 xi0) stored modulo 2 pi on a grid with
  // no periodic axis: it wraps between interior nodes
  constexpr Real TwoPi = 2 * std::numbers::pi;
  const auto theta = [](Real x, Real y) {
    return TwoPi * (0.7 + 0.6 * y + 0.1 * x);
  };
  const auto grid = pcms::test::MakeUnitSquareGrid(8);
  const auto position = pcms::test::MakePositionFunction(
    pcms::CreateLagrangeSpace(grid, 1, 2),
    [&](const Real* xi, Real* u) {
      u[0] = xi[0];
      u[1] = std::fmod(theta(xi[0], xi[1]), TwoPi);
    },
    pcms::csys::CreateParameterSystem("u", 2, {{1, 0.0, TwoPi}}));
  const auto points = RandomPoints(40, {0.0, 0.0});
  const auto mapped = MapPoints(position, grid, points, 2);
  for (std::size_t p = 0; p < points.size() / 2; ++p) {
    CAPTURE(p);
    const Real x = points[2 * p];
    const Real y = points[2 * p + 1];
    CHECK_THAT(std::remainder(mapped.x(p, 1) - theta(x, y), TwoPi),
               WithinAbs(0.0, 1e-9));
    const std::array<Real, 4> expected = {1.0, 0.0, 0.1 * TwoPi, 0.6 * TwoPi};
    for (int k = 0; k < 4; ++k) {
      CHECK_THAT(mapped.jacobian(p, k), WithinAbs(expected[k], 1e-9));
    }
  }
}
