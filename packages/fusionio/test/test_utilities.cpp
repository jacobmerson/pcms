// Fusion-IO 1.x's utilities against Fusion-IO 2's, on the equilibrium
// poloidal flux of the example C1.h5. Both evaluate the same 1.x fields (the
// source is opened with Evaluation::FusionIO) and the batched functions run
// 1.x's iterations, so they take the same steps up to the round-off of the
// gradient's (1/R) d/dphi scaling. Where a batched function starts a search
// from another guess than 1.x's sequential loop does, the tolerance follows
// from the search's own.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fusion_io.h>
#include <fusionio/fusionio.hpp>
#include <fusionio/utilities.hpp>
#include "fusion_io_source.hpp"
#include <isosurface.h>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::Point;
using fusionio::Real;
using fusionio::SearchResult;

namespace
{

const std::string M3DC1File =
  std::string(FUSIONIO_TEST_DATA_DIR) + "/m3dc1/C1.h5";

/// The equilibrium poloidal flux of C1.h5, through Fusion-IO 2 and 1.x.
struct Flux
{
  std::shared_ptr<fusionio::Source> source =
    fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
  pcms::Function<Real> psi =
    source->CreateFunction(fusionio::Quantity::PoloidalFlux, {.timeslice = -1});
  fusionio::detail::FusionIOSourcePtr source1x = Open1x();
  fusionio::detail::FusionIOFieldPtr psi1x = Field1x(FIO_POLOIDAL_FLUX);

  static fusionio::detail::FusionIOSourcePtr Open1x()
  {
    fio_source* opened = nullptr;
    REQUIRE(fio_open_source(&opened, FIO_M3DC1_SOURCE, M3DC1File.c_str()) ==
            FIO_SUCCESS);
    return fusionio::detail::FusionIOSourcePtr(opened);
  }

  [[nodiscard]] fusionio::detail::FusionIOFieldPtr Field1x(
    field_type type) const
  {
    fio_option_list options;
    source1x->get_field_options(&options);
    options.set_option(FIO_TIMESLICE, -1);
    fio_field* field = nullptr;
    REQUIRE(source1x->get_field(type, &field, &options) == FIO_SUCCESS);
    return fusionio::detail::FusionIOFieldPtr(field);
  }

  /// 1.x's psi at x
  [[nodiscard]] Real Psi1x(Point x) const
  {
    Real v = 0.;
    REQUIRE(psi1x->eval(x.data(), &v) == FIO_SUCCESS);
    return v;
  }
};

SearchResult Result1x(int result)
{
  return result == FIO_SUCCESS         ? SearchResult::Success
         : result == FIO_OUT_OF_BOUNDS ? SearchResult::OutOfBounds
                                       : SearchResult::Diverged;
}

void RequirePoint(const Point& actual, const Point& expected, Real tol)
{
  for (int c = 0; c < 3; ++c) {
    CAPTURE(c);
    REQUIRE_THAT(actual[c], WithinAbs(expected[c], tol));
  }
}

} // namespace

TEST_CASE("fusionio BoundingBox is get_bounding_box")
{
  Flux flux;
  Point min{}, max{};
  REQUIRE(flux.psi1x->get_bounding_box(min.data(), max.data()) == FIO_SUCCESS);
  // the same mesh extent, computed by 1.x
  const auto box = flux.source->BoundingBox();
  REQUIRE(box.has_value());
  RequirePoint(box->min, min, 0.);
  RequirePoint(box->max, max, 0.);
  const auto native = fusionio::Open(fusionio::Format::M3DC1, M3DC1File,
                                     fusionio::Evaluation::Native);
  RequirePoint(native->BoundingBox()->min, min, 0.);
  RequirePoint(native->BoundingBox()->max, max, 0.);
  // 1.x has none for geqdsk files
  const auto geqdsk = fusionio::Open(fusionio::Format::GEQDSK,
                                     std::string(FUSIONIO_TEST_DATA_DIR) +
                                       "/geqdsk/g158115.04701");
  REQUIRE_FALSE(geqdsk->BoundingBox().has_value());
}

TEST_CASE("fusionio::FindMax matches fio_find_max")
{
  Flux flux;
  const Point norm{0., 1., 0.};
  const Real tol = 1e-12;
  std::vector<Point> x = {{1.65, 0., 0.05}, {1.75, 0.5, -0.1}, {1.6, 1., 0.2}};
  std::vector<Real> val(x.size(), 0.);
  const auto expected_x = x;
  const auto result = fusionio::FindMax(flux.psi, val, x, tol, 0.1, 2, norm);
  for (size_t k = 0; k < x.size(); ++k) {
    CAPTURE(k);
    Point x1 = expected_x[k];
    Real val1 = 0.;
    const int result1 = fio_find_max(flux.psi1x.get(), &val1, x1.data(), tol,
                                     0.1, 2, norm.data());
    REQUIRE(result[k] == Result1x(result1));
    REQUIRE(result[k] == SearchResult::Success);
    // The values agree to round-off (the evaluations may take a point on an
    // element boundary from either element), and the search stops once psi
    // changes by less than tol in a step, within sqrt(tol / psi'') of the
    // axis (about 1e-6 m here): the two stop that close to each other.
    RequirePoint(x[k], x1, 1e-6);
    REQUIRE_THAT(val[k], WithinRel(val1, 1e-12));
  }
}

TEST_CASE("fusionio::FindVal matches fio_find_val")
{
  Flux flux;
  const Real tol = 1e-10 * std::abs(flux.Psi1x({1.9, 0., 0.}));
  const Real val = flux.Psi1x({1.9, 0., 0.});
  // along R (dim 1) from both sides, and down the gradient in a phi plane
  for (const int dim : {1, 2}) {
    CAPTURE(dim);
    const Point norm = dim == 1 ? Point{1., 0., 0.} : Point{0., 1., 0.};
    std::vector<Point> x = {{1.8, 0., 0.}, {2.0, 0.3, 0.}, {1.85, 2., 0.1}};
    const auto start = x;
    const auto result =
      fusionio::FindVal(flux.psi, val, x, tol, 0.1, dim, norm);
    for (size_t k = 0; k < x.size(); ++k) {
      CAPTURE(k);
      Point x1 = start[k];
      const int result1 = fio_find_val(flux.psi1x.get(), val, x1.data(), tol,
                                       0.1, dim, norm.data());
      REQUIRE(result[k] == Result1x(result1));
      REQUIRE(result[k] == SearchResult::Success);
      RequirePoint(x[k], x1, 1e-10);
    }
  }
  // a guess outside the domain
  std::vector<Point> outside = {{10., 0., 0.}};
  REQUIRE(fusionio::FindVal(flux.psi, val, outside, tol, 0.1, 1,
                            {1., 0., 0.})[0] == SearchResult::OutOfBounds);
}

TEST_CASE("fusionio::FindValOnLine matches find_val_on_line")
{
  Flux flux;
  const Real val = flux.Psi1x({1.9, 0., 0.});
  const Real tol = 1e-10 * std::abs(val);
  const std::vector<Point> x0 = {{1.7, 0., 0.}, {1.7, 0., 0.}};
  const std::vector<Point> x1 = {{2.1, 0., 0.}, {2.0, 0., 0.3}};
  std::vector<Point> x = {{1.95, 0., 0.}, {1.8, 0., 0.1}};
  const auto start = x;
  const auto result = fusionio::FindValOnLine(flux.psi, val, x0, x1, x, tol);
  for (size_t k = 0; k < x.size(); ++k) {
    CAPTURE(k);
    Point x_1x = start[k];
    const int result1 = flux.psi1x->find_val_on_line(
      val, x0[k].data(), x1[k].data(), x_1x.data(), nullptr, tol);
    REQUIRE(result[k] == Result1x(result1));
    REQUIRE(result[k] == SearchResult::Success);
    RequirePoint(x[k], x_1x, 1e-10);
  }
}

// 1.x's contour (fio_isosurface_2d) through `guess` at psi = val
std::vector<Point> Contour1x(Flux& flux, Real val, Point guess, Real dl,
                             Real tol)
{
  const Point norm{0., 1., 0.};
  int n = 0;
  double** path = nullptr;
  const int result =
    fio_isosurface_2d(flux.psi1x.get(), val, guess.data(), nullptr, norm.data(),
                      nullptr, dl, tol, 0.1, &n, &path);
  REQUIRE(result == FIO_SUCCESS);
  std::vector<Point> points(n);
  for (int i = 0; i < n; ++i) {
    points[i] = {path[0][i], path[1][i], path[2][i]};
  }
  for (int c = 0; c < 3; ++c) {
    delete[] path[c];
  }
  delete[] path;
  return points;
}

// 1.x's path arrays, path[c][i], for a range of points
struct Arrays1x
{
  explicit Arrays1x(size_t n) : values(3, std::vector<double>(n))
  {
    for (int c = 0; c < 3; ++c) {
      rows[c] = values[c].data();
    }
  }
  explicit Arrays1x(const std::vector<Point>& points) : Arrays1x(points.size())
  {
    for (size_t i = 0; i < points.size(); ++i) {
      for (int c = 0; c < 3; ++c) {
        values[c][i] = points[i][c];
      }
    }
  }
  [[nodiscard]] Point At(size_t i) const
  {
    return {values[0][i], values[1][i], values[2][i]};
  }
  std::vector<std::vector<double>> values;
  double* rows[3] = {};
};

TEST_CASE("fusionio::GridifyLoop and GridifySurface match 1.x")
{
  Flux flux;
  const Real val = flux.Psi1x({1.9, 0., 0.});
  const auto contour = Contour1x(flux, val, {1.9, 0., 0.}, 0.01, 1e-10);
  const Point axis{1.7537, 0., 0.0172};
  constexpr int M = 64;
  using fusionio::LoopParameter;
  // a toroidal circle for the toroidal angle
  std::vector<Point> circle;
  for (int i = 0; i < 50; ++i) {
    circle.push_back({1.9, 2 * std::numbers::pi * i / 50, 0.});
  }
  // 1.x's angle parameters fail on closed loops (the loop's closing point
  // takes the first point's angle, so the angle is not monotonic): both
  // report it
  for (const auto& [param, param1x, loop] :
       {std::tuple{LoopParameter::ArcLength, 0, contour},
        std::tuple{LoopParameter::PoloidalAngle, 1, contour},
        std::tuple{LoopParameter::ToroidalAngle, 2, circle}}) {
    CAPTURE(param1x);
    std::vector<Point> path(M);
    std::vector<Real> theta(M);
    const auto result = fusionio::GridifyLoop(loop, axis, path, theta, param);
    Arrays1x loop1x(loop), path1x(M);
    std::vector<double> theta1x(M);
    const int result1x =
      fio_gridify_loop(static_cast<int>(loop.size()), loop1x.rows, axis.data(),
                       M, path1x.rows, theta1x.data(), param1x);
    REQUIRE(result == Result1x(result1x));
    if (result != SearchResult::Success) {
      continue;
    }
    // the same arithmetic
    for (int i = 0; i < M; ++i) {
      CAPTURE(i);
      RequirePoint(path[i], path1x.At(i), 1e-14);
      REQUIRE_THAT(theta[i], WithinAbs(theta1x[i], 1e-14));
    }
  }
  // the contour in two planes of phi, one after the other
  std::vector<Point> surface0 = contour;
  for (Point p : contour) {
    p[1] = 1.;
    surface0.push_back(p);
  }
  std::vector<Point> surface(2 * M);
  std::vector<Real> phi(2), theta(M);
  REQUIRE(fusionio::GridifySurface(surface0, axis, 2, M, surface, phi, theta) ==
          SearchResult::Success);
  Arrays1x surface0_1x(surface0), surface1x(2 * M);
  std::vector<double> phi1x(2), theta1x(M);
  REQUIRE(fio_gridify_surface(
            static_cast<int>(surface0.size()), surface0_1x.rows, axis.data(), 2,
            M, surface1x.rows, phi1x.data(), theta1x.data()) == FIO_SUCCESS);
  for (int i = 0; i < 2 * M; ++i) {
    CAPTURE(i);
    RequirePoint(surface[i], surface1x.At(i), 1e-14);
  }
  REQUIRE(phi == std::vector<Real>(phi1x.begin(), phi1x.end()));
}

// fio_isosurface as isosurface.cpp defines it (isosurface.h declares a
// double where the definition takes the number of planes)
int fio_isosurface(fio_field* f, double val, const double* guess,
                   const double* enclose, double dl, double tol,
                   double max_step, int nphi, int* n, double*** path,
                   fio_hint h);

// 1.x's path arrays (path[c][i], allocated by 1.x) as points, freed
std::vector<Point> TakePath1x(double** path, int n)
{
  std::vector<Point> points(n);
  for (int i = 0; i < n; ++i) {
    points[i] = {path[0][i], path[1][i], path[2][i]};
  }
  for (int c = 0; c < 3; ++c) {
    delete[] path[c];
  }
  delete[] path;
  return points;
}

void RequirePath(const std::vector<Point>& actual,
                 const std::vector<Point>& expected, Real tol)
{
  REQUIRE(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CAPTURE(i);
    RequirePoint(actual[i], expected[i], tol);
  }
}

TEST_CASE("fusionio isosurfaces match 1.x's")
{
  Flux flux;
  const Real val = flux.Psi1x({1.9, 0., 0.});
  const Real tol = 1e-10 * std::abs(val);
  const Point axis{1.7537, 0., 0.0172};
  // The R and Z partials are 1.x's, and the phi one is zero (the field is
  // axisymmetric), so the steps are 1.x's to the last bit but for the
  // compiler's contractions.
  constexpr Real Exact = 1e-12;

  SECTION("Isosurface2D: one contour per guess, together")
  {
    const std::vector<Point> guess = {{1.9, 0., 0.}, {1.95, 1., 0.05}};
    const Point norm{0., 1., 0.};
    const auto paths = fusionio::Isosurface2D(flux.psi, val, guess, norm,
                                              std::nullopt, 0.02, tol, 0.1);
    for (size_t k = 0; k < guess.size(); ++k) {
      CAPTURE(k);
      REQUIRE(paths[k].result == SearchResult::Success);
      RequirePath(paths[k].points, Contour1x(flux, val, guess[k], 0.02, tol),
                  Exact);
    }
  }

  SECTION("Isosurface: loops in two planes")
  {
    const std::vector<Point> guess = {{1.9, 0., 0.}};
    const auto paths =
      fusionio::Isosurface(flux.psi, val, guess, axis, 0.02, tol, 0.1, 2);
    int n = 0;
    double** path = nullptr;
    REQUIRE(fio_isosurface(flux.psi1x.get(), val, guess[0].data(), axis.data(),
                           0.02, tol, 0.1, 2, &n, &path,
                           nullptr) == FIO_SUCCESS);
    REQUIRE(paths[0].result == SearchResult::Success);
    RequirePath(paths[0].points, TakePath1x(path, n), Exact);
  }

  SECTION("GeomIsosurface: a grid of geometric angles")
  {
    constexpr int NPhi = 2, NTheta = 16;
    const Point guess{1.9, 0., 0.};
    const auto path = fusionio::GeomIsosurface(flux.psi, val, guess, axis, tol,
                                               0.1, NPhi, NTheta);
    REQUIRE(path.result == SearchResult::Success);
    Arrays1x path1x(NPhi * NTheta);
    double** rows = path1x.rows;
    REQUIRE(fio_geom_isosurface(flux.psi1x.get(), val, guess.data(),
                                axis.data(), tol, 0.1, NPhi, NTheta,
                                &rows) == FIO_SUCCESS);
    // the first plane searches as 1.x does
    for (int j = 0; j < NTheta; ++j) {
      CAPTURE(j);
      RequirePoint(path.points[j], path1x.At(j), Exact);
    }
    // 1.x starts the second plane from the first's last distance, this from
    // the guess's: other points of the same surface
    for (int j = NTheta; j < NPhi * NTheta; ++j) {
      CAPTURE(j);
      REQUIRE_THAT(flux.Psi1x(path.points[j]), WithinAbs(val, tol));
      REQUIRE(path.points[j][1] == path1x.At(j)[1]);
    }
  }
}

TEST_CASE("fusionio::GriddedIsosurface matches fio_gridded_isosurface")
{
  Flux flux;
  const Real val = flux.Psi1x({1.9, 0., 0.});
  const Real tol = 1e-10 * std::abs(val);
  constexpr int NPhi = 3, NTheta = 32;
  const Point guess{1.9, 0., 0.};
  const Point axis{1.75, 0., 0.};
  const auto surface = fusionio::GriddedIsosurface(
    flux.psi, val, guess, axis, 0.02, tol, 0.1, NPhi, NTheta);
  REQUIRE(surface.result == SearchResult::Success);

  Arrays1x axis1x(std::vector<Point>(NPhi, axis)), path1x(NPhi * NTheta);
  std::vector<double> phi1x(NPhi), theta1x(NTheta);
  REQUIRE(fio_gridded_isosurface(flux.psi1x.get(), val, guess.data(),
                                 axis1x.rows, 0.02, 0.02, tol, 0.1, NPhi,
                                 NTheta, phi1x.data(), theta1x.data(),
                                 path1x.rows, nullptr) == FIO_SUCCESS);
  REQUIRE(surface.phi == std::vector<Real>(phi1x.begin(), phi1x.end()));
  REQUIRE(surface.theta == std::vector<Real>(theta1x.begin(), theta1x.end()));
  // The first plane is 1.x's search. 1.x starts each later plane's axis and
  // contour search from the previous plane's results, this from the
  // arguments. The axis search stops once psi changes by less than tol in a
  // step, within sqrt(tol / psi'') of the axis (about 1e-6 m here), and the
  // contour starts level with the axis, so the gridded points move as much.
  for (int i = 0; i < NPhi; ++i) {
    CAPTURE(i);
    const Real tolerance = i == 0 ? 1e-12 : 1e-5;
    RequirePoint(surface.axis[i], axis1x.At(i), tolerance);
    for (int j = 0; j < NTheta; ++j) {
      CAPTURE(j);
      RequirePoint(surface.path[i * NTheta + j], path1x.At(i * NTheta + j),
                   tolerance);
    }
  }
}

TEST_CASE("fusionio hybrid searches match 1.x's")
{
  Flux flux;
  const Real val = flux.Psi1x({1.9, 0., 0.});
  const Real tol = 1e-10 * std::abs(val);
  const Point axis{1.7537, 0., 0.0172};

  SECTION("FindRhoHybrid: rays together")
  {
    std::vector<Real> ct, st, rho;
    for (int j = 0; j < 8; ++j) {
      ct.push_back(std::cos(0.8 * j));
      st.push_back(std::sin(0.8 * j));
      rho.push_back(0.05 + 0.02 * j);
    }
    const auto start = rho;
    const auto result =
      fusionio::FindRhoHybrid(flux.psi, val, rho, ct, st, 0.5, axis, tol);
    for (size_t j = 0; j < rho.size(); ++j) {
      CAPTURE(j);
      double rho1x = start[j];
      const int result1x =
        fio_find_rho_hybrid(flux.psi1x.get(), val, rho1x, ct[j], st[j], 0.5,
                            axis.data(), tol, nullptr);
      REQUIRE(result[j] == Result1x(result1x));
      REQUIRE_THAT(rho[j], WithinAbs(rho1x, 1e-12));
    }
  }

  SECTION("GriddedIsosurfaceHybrid")
  {
    constexpr int NPhi = 2, NTheta = 16;
    const Point guess{1.9, 0., 0.};
    const auto surface = fusionio::GriddedIsosurfaceHybrid(
      flux.psi, val, guess, axis, tol, 0.1, NPhi, NTheta);
    REQUIRE(surface.result == SearchResult::Success);
    Arrays1x axis1x(std::vector<Point>(NPhi, axis)), path1x(NPhi * NTheta);
    std::vector<double> phi1x(NPhi), theta1x(NTheta);
    REQUIRE(fio_gridded_isosurface_hybrid(
              flux.psi1x.get(), val, guess.data(), axis1x.rows, 0.02, 0.02, tol,
              0.1, NPhi, NTheta, phi1x.data(), theta1x.data(), path1x.rows,
              nullptr) == FIO_SUCCESS);
    // The axes are 1.x's search. 1.x starts each angle from the radius just
    // found at the previous angle, this from the previous plane's: both
    // stop within the searches' tolerances (|psi - val| < tol or a Newton
    // step below 1e-8 m).
    for (int i = 0; i < NPhi; ++i) {
      CAPTURE(i);
      RequirePoint(surface.axis[i], axis1x.At(i), 1e-12);
    }
    for (int k = 0; k < NPhi * NTheta; ++k) {
      CAPTURE(k);
      RequirePoint(surface.path[k], path1x.At(k), 1e-7);
    }
  }
}
