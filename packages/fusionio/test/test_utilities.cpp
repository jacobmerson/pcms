// Fusion-IO 1.x's utilities against Fusion-IO 2's, on the equilibrium
// poloidal flux of the example C1.h5. Both evaluate the same 1.x fields (the
// source is opened with Evaluation::FusionIO).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fusionio/fusionio.hpp>
#include "fusion_io_source.hpp"
#include <memory>
#include <string>

using Catch::Matchers::WithinAbs;
using fusionio::Point;
using fusionio::Real;

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
};

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
