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
