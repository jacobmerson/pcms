#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <fusionio/fusionio.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include "test_utils.hpp"
#include <algorithm>
#include <numbers>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::Quantity;
using fusionio::Real;
using fusionio::test::EvaluateAt;

namespace
{

const std::string GeqdskFile =
  std::string(FUSIONIO_TEST_DATA_DIR) + "/geqdsk/g158115.04701";
const std::string M3DC1File =
  std::string(FUSIONIO_TEST_DATA_DIR) + "/m3dc1/C1.h5";

// Points inside the example M3D-C1 domain, (R, phi, Z) row-major.
const std::vector<Real> M3DC1Points = {1.6, 0.1, 0.2, 2.0, 2.5, -0.4};

int ComponentsOf(const fusionio::Source& source, Quantity quantity)
{
  const auto quantities = source.Quantities();
  const auto found =
    std::ranges::find(quantities, quantity, &fusionio::QuantityInfo::quantity);
  return found == quantities.end() ? 0 : found->num_components;
}

} // namespace

TEST_CASE("fusionio quantity names are Fusion-IO's")
{
  REQUIRE(fusionio::Name(Quantity::MagneticField) == "magnetic field");
  REQUIRE(fusionio::Name(Quantity::PoloidalFluxNorm) ==
          "normalized poloidal flux");
  REQUIRE(fusionio::Name(Quantity::BootstrapCurrent) == "JpdotB");
}

TEST_CASE("fusionio quantities are primal, with units where known")
{
  const auto b = fusionio::Metadata(Quantity::MagneticField);
  REQUIRE(b.name == "magnetic field");
  REQUIRE(b.duality == pcms::Duality::Primal);
  const auto b_units =
    fusionio::QuantityBasis(Quantity::MagneticField,
                            pcms::csys::CreateCylindricalRThetaZ())
      .units;
  REQUIRE(b_units.has_value());
  REQUIRE(b_units->ToString() == pcms::Units::Parse("T").ToString());
  REQUIRE(fusionio::Metadata(Quantity::PoloidalFlux).duality ==
          pcms::Duality::Primal);
  REQUIRE_FALSE(fusionio::QuantityBasis(Quantity::PoloidalFlux, nullptr).units);
}

TEST_CASE("fusionio::Open describes an M3D-C1 source")
{
  const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);

  REQUIRE(ComponentsOf(*source, Quantity::MagneticField) == 3);
  REQUIRE(ComponentsOf(*source, Quantity::PoloidalFlux) == 1);
  REQUIRE(source->Times().size() == 2);
  REQUIRE_THAT(
    source->GetGeometry()->GetCoordinateSystem()->PeriodicAxes().front().length,
    WithinRel(2 * std::numbers::pi, 1e-14));

  // The magnetic axis lies inside the domain.
  const auto axis_r =
    source->EvaluateSeries(fusionio::Series::MagneticAxisR, source->Times()[0]);
  REQUIRE(axis_r.has_value());
  REQUIRE(*axis_r > 1.0);
  REQUIRE(*axis_r < 2.5);
}

TEST_CASE("fusionio::Open functions share the source's geometry")
{
  const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
  const fusionio::Options options{.timeslice = 1,
                                  .part = fusionio::Part::Perturbed};
  const auto b = source->CreateFunction(Quantity::MagneticField, options);
  const auto psi = source->CreateFunction(Quantity::PoloidalFlux, options);
  const auto p = source->CreateFunction(Quantity::TotalPressure, options);
  REQUIRE(b.GetName() == "magnetic field");
  REQUIRE(b.GetData().GetValueBasis().units->ToString() ==
          pcms::Units::Parse("T").ToString());
  REQUIRE(b.GetMetadata().duality == pcms::Duality::Primal);
  REQUIRE(b.GetLayout().GetNumComponents() == 3);
  REQUIRE(b.GetSpace().GetGeometry() == source->GetGeometry());
  REQUIRE(psi.GetSpace().GetGeometry() == source->GetGeometry());
  // a space per representation: the scalars share one
  REQUIRE(&psi.GetSpace() == &p.GetSpace());
  REQUIRE(&psi.GetSpace() != &b.GetSpace());
  REQUIRE(psi.GetSpace().Supports(pcms::DifferentialOperator::Gradient));

  // One binding serves every quantity of the source, and evaluates as a
  // binding of each function's own does.
  pcms::QueryPoints query(source->GetGeometry());
  const auto coords = fusionio::test::CreateDeviceCoordinateView(
    M3DC1Points, source->GetGeometry()->GetCoordinateSystem(), 3);
  query.Locate(coords.coordinate_view);
  for (const auto* f : {&b, &psi}) {
    const int n = f->GetLayout().GetNumComponents();
    Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", 2, n);
    query.Evaluate(*f, fusionio::test::TagLike(*f, out));
    const auto shared =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
    const auto separate = EvaluateAt(*f, M3DC1Points);
    for (size_t i = 0; i < 2; ++i) {
      for (int c = 0; c < n; ++c) {
        REQUIRE(shared(i, c) == separate(i, c));
      }
    }
  }
}

TEST_CASE("fusionio::Open reports derivatives Fusion-IO provides")
{
  const auto geqdsk = fusionio::Open(fusionio::Format::GEQDSK, GeqdskFile);
  REQUIRE(geqdsk->CreateFunction(Quantity::PoloidalFlux)
            .GetSpace()
            .Supports(pcms::DifferentialOperator::Gradient));
  const auto p = geqdsk->CreateFunction(Quantity::TotalPressure);
  REQUIRE_FALSE(p.GetSpace().Supports(pcms::DifferentialOperator::Gradient));
  REQUIRE_THROWS_WITH(fusionio::test::EvaluateGradientAt(p, {1.8, 0.0, 0.1}),
                      ContainsSubstring("Gradient is not supported"));
}

TEST_CASE("fusionio phase is a toroidal shift in radians")
{
  // Time slice 1 holds an n = 2 perturbation, so the field depends on phi.
  const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
  const Real phase = 0.3;
  const auto shifted = source->CreateFunction(
    Quantity::PoloidalFlux,
    {.timeslice = 1, .part = fusionio::Part::Perturbed, .phase = phase});
  const auto unshifted =
    source->CreateFunction(Quantity::PoloidalFlux,
                           {.timeslice = 1, .part = fusionio::Part::Perturbed});

  const auto at_phi = EvaluateAt(shifted, {1.6, 0.5, 0.2});
  const auto at_phi_minus_phase =
    EvaluateAt(unshifted, {1.6, 0.5 - phase, 0.2});
  const auto at_phi_unshifted = EvaluateAt(unshifted, {1.6, 0.5, 0.2});
  REQUIRE_THAT(at_phi(0, 0), WithinRel(at_phi_minus_phase(0, 0), 1e-12));
  REQUIRE(at_phi(0, 0) != at_phi_unshifted(0, 0));
}

TEST_CASE("fusionio::Open describes a geqdsk source")
{
  const auto source = fusionio::Open(fusionio::Format::GEQDSK, GeqdskFile);

  REQUIRE(source->Times().empty());
  REQUIRE(ComponentsOf(*source, Quantity::MagneticField) == 3);
  // The file's header gives rmaxis = 1.73434.
  const auto axis_r =
    source->EvaluateSeries(fusionio::Series::MagneticAxisR, 0.0);
  REQUIRE(axis_r.has_value());
  REQUIRE_THAT(*axis_r, WithinAbs(1.73434, 1e-5));
}

TEST_CASE("fusionio reports quantities a source does not provide")
{
  const auto source = fusionio::Open(fusionio::Format::GEQDSK, GeqdskFile);
  REQUIRE(ComponentsOf(*source, Quantity::Viscosity) == 0);
  REQUIRE_THROWS_WITH(source->CreateFunction(Quantity::Viscosity),
                      ContainsSubstring("viscosity"));
}

TEST_CASE("fusionio functions fill points outside the domain on request")
{
  const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
  const auto b = source->CreateFunction(Quantity::MagneticField);
  const auto& space = b.GetSpace();
  // The middle point is far outside the domain.
  const std::vector<Real> points = {1.6, 0.1, 0.2, 50.0, 0.0,
                                    0.0, 2.0, 2.5, -0.4};
  const auto coords = fusionio::test::CreateDeviceCoordinateView(
    points, space.GetCoordinateSystem(), 3);
  constexpr Real Fill = -7.0;

  SECTION("fill")
  {
    const auto evaluator =
      space.CreatePointEvaluator<Real>(space.GetGeometry()->Bind(
        coords.coordinate_view, {pcms::OutOfBoundsMode::FILL, Fill}));
    const auto filled = Kokkos::create_mirror_view_and_copy(
      pcms::HostMemorySpace(), evaluator->FilledPoints());
    REQUIRE(filled.extent(0) == 1);
    REQUIRE(filled(0) == 1);

    Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", 3, 3);
    evaluator->Evaluate(b, fusionio::test::TagLike<Real>(b, out));
    const auto values =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
    const auto expected = EvaluateAt(b, {1.6, 0.1, 0.2, 2.0, 2.5, -0.4});
    for (size_t c = 0; c < 3; ++c) {
      REQUIRE(values(0, c) == expected(0, c));
      REQUIRE(values(1, c) == Fill);
      REQUIRE(values(2, c) == expected(1, c));
    }
  }

  SECTION("M3D-C1 sources locate at bind, so ERROR throws there")
  {
    REQUIRE_THROWS_WITH(space.GetGeometry()->Bind(coords.coordinate_view),
                        ContainsSubstring("outside the source's domain"));
  }

  SECTION("nearest boundary is not supported")
  {
    REQUIRE_THROWS_WITH(
      space.GetGeometry()->Bind(coords.coordinate_view,
                                {pcms::OutOfBoundsMode::NEAREST_BOUNDARY}),
      ContainsSubstring("NEAREST_BOUNDARY"));
  }
}
