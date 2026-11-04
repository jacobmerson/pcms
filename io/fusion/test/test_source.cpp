#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <fusion_io/fusion_io.hpp>
#include "fusion_io_space.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <numbers>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fio::Quantity;
using fio::Real;
using pcms::test::EvaluateAt;

namespace
{

const std::string GeqdskFile =
  std::string(PCMS_TEST_FUSIONIO_DIR) + "/geqdsk/g158115.04701";
const std::string M3DC1File =
  std::string(PCMS_TEST_FUSIONIO_DIR) + "/m3dc1/C1.h5";

// Points inside the example M3D-C1 domain, (R, phi, Z) row-major.
const std::vector<Real> M3DC1Points = {1.6, 0.1, 0.2, 2.0, 2.5, -0.4};

int ComponentsOf(const fio::Source& source, Quantity quantity)
{
  const auto quantities = source.Quantities();
  const auto found =
    std::ranges::find(quantities, quantity, &fio::QuantityInfo::quantity);
  return found == quantities.end() ? 0 : found->num_components;
}

} // namespace

TEST_CASE("fio quantity names are Fusion-IO's")
{
  REQUIRE(fio::Name(Quantity::MagneticField) == "magnetic field");
  REQUIRE(fio::Name(Quantity::PoloidalFluxNorm) == "normalized poloidal flux");
  REQUIRE(fio::Name(Quantity::BootstrapCurrent) == "JpdotB");
}

TEST_CASE("fio::Open describes an M3D-C1 source")
{
  const auto source = fio::Open(fio::Format::M3DC1, M3DC1File);

  REQUIRE(ComponentsOf(*source, Quantity::MagneticField) == 3);
  REQUIRE(ComponentsOf(*source, Quantity::PoloidalFlux) == 1);
  REQUIRE(source->Times().size() == 2);
  REQUIRE_THAT(source->Period(), WithinRel(2 * std::numbers::pi, 1e-14));

  // The magnetic axis lies inside the domain.
  const auto axis_r =
    source->EvaluateSeries(fio::Series::MagneticAxisR, source->Times()[0]);
  REQUIRE(axis_r.has_value());
  REQUIRE(*axis_r > 1.0);
  REQUIRE(*axis_r < 2.5);
}

TEST_CASE("fio::Open functions match the Fusion-IO function space")
{
  const auto source = fio::Open(fio::Format::M3DC1, M3DC1File);
  const fio::Options options{.timeslice = 1, .part = fio::Part::Perturbed};
  const auto b = source->CreateFunction(Quantity::MagneticField, options);
  REQUIRE(b.GetName() == "magnetic field");
  REQUIRE(b.GetLayout().GetNumComponents() == 3);

  const auto space = fio::detail::FusionIOFunctionSpace::Create(
    fio::detail::OpenFusionIOSource(fio::Format::M3DC1, M3DC1File), 3);
  const auto expected = space->CreateFunction("b", "magnetic field", options);

  const auto actual_values = EvaluateAt(b, M3DC1Points);
  const auto expected_values = EvaluateAt(expected, M3DC1Points);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t c = 0; c < 3; ++c) {
      REQUIRE(actual_values(i, c) == expected_values(i, c));
    }
  }
}

TEST_CASE("fio phase is a toroidal shift in radians")
{
  // Time slice 1 holds an n = 2 perturbation, so the field depends on phi.
  const auto source = fio::Open(fio::Format::M3DC1, M3DC1File);
  const Real phase = 0.3;
  const auto shifted = source->CreateFunction(
    Quantity::PoloidalFlux,
    {.timeslice = 1, .part = fio::Part::Perturbed, .phase = phase});
  const auto unshifted = source->CreateFunction(
    Quantity::PoloidalFlux, {.timeslice = 1, .part = fio::Part::Perturbed});

  const auto at_phi = EvaluateAt(shifted, {1.6, 0.5, 0.2});
  const auto at_phi_minus_phase =
    EvaluateAt(unshifted, {1.6, 0.5 - phase, 0.2});
  const auto at_phi_unshifted = EvaluateAt(unshifted, {1.6, 0.5, 0.2});
  REQUIRE_THAT(at_phi(0, 0), WithinRel(at_phi_minus_phase(0, 0), 1e-12));
  REQUIRE(at_phi(0, 0) != at_phi_unshifted(0, 0));
}

TEST_CASE("fio::Open describes a geqdsk source")
{
  const auto source = fio::Open(fio::Format::GEQDSK, GeqdskFile);

  REQUIRE(source->Times().empty());
  REQUIRE(ComponentsOf(*source, Quantity::MagneticField) == 3);
  // The file's header gives rmaxis = 1.73434.
  const auto axis_r = source->EvaluateSeries(fio::Series::MagneticAxisR, 0.0);
  REQUIRE(axis_r.has_value());
  REQUIRE_THAT(*axis_r, WithinAbs(1.73434, 1e-5));
}

TEST_CASE("fio reports quantities a source does not provide")
{
  const auto source = fio::Open(fio::Format::GEQDSK, GeqdskFile);
  REQUIRE(ComponentsOf(*source, Quantity::Viscosity) == 0);
  REQUIRE_THROWS_WITH(source->CreateFunction(Quantity::Viscosity),
                      ContainsSubstring("viscosity"));
}

TEST_CASE("fio functions fill points outside the domain on request")
{
  const auto source = fio::Open(fio::Format::M3DC1, M3DC1File);
  const auto b = source->CreateFunction(Quantity::MagneticField);
  const auto& space = b.GetSpace();
  // The middle point is far outside the domain.
  const std::vector<Real> points = {1.6, 0.1, 0.2, 50.0, 0.0,
                                    0.0, 2.0, 2.5, -0.4};
  const auto coords = pcms::test::CreateDeviceCoordinateView(
    points, space.GetCoordinateSystem(), 3);
  constexpr Real Fill = -7.0;

  SECTION("fill")
  {
    const auto evaluator = space.CreatePointEvaluator<Real>(
      space.Bind(coords.coordinate_view, {pcms::OutOfBoundsMode::FILL, Fill}));
    const auto filled = Kokkos::create_mirror_view_and_copy(
      pcms::HostMemorySpace(), evaluator->FilledPoints());
    REQUIRE(filled.extent(0) == 1);
    REQUIRE(filled(0) == 1);

    Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", 3, 3);
    evaluator->Evaluate(b, pcms::test::TagLike<Real>(b, out));
    const auto values =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
    const auto expected = EvaluateAt(b, {1.6, 0.1, 0.2, 2.0, 2.5, -0.4});
    for (size_t c = 0; c < 3; ++c) {
      REQUIRE(values(0, c) == expected(0, c));
      REQUIRE(values(1, c) == Fill);
      REQUIRE(values(2, c) == expected(1, c));
    }
  }

  SECTION("nearest boundary is not supported")
  {
    REQUIRE_THROWS_WITH(space.Bind(coords.coordinate_view,
                                   {pcms::OutOfBoundsMode::NEAREST_BOUNDARY}),
                        ContainsSubstring("NEAREST_BOUNDARY"));
  }
}
