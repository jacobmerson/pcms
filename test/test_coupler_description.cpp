#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "description.hpp"
#include "embedding_test_utils.hpp"
#include "pcms/field/embedding.hpp"
#include "pcms/field/periodic_axis.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include "pcms/utility/coordinate_systems/cylindrical.hpp"
#include "pcms/utility/coordinate_systems/parameter.hpp"
#include "pcms/utility/detail/coordinate_system.hpp"
#include <numbers>

namespace detail = pcms::detail;
using pcms::Real;

namespace
{

std::string UnitsText(const std::optional<pcms::Units>& units)
{
  return units ? units->ToString() : "none";
}

} // namespace

TEST_CASE("a setup description reads back what was written")
{
  auto description = detail::CreateDescription();
  description["function.0.name"] = "density = rho";
  description["geometry.0.bot_left"] =
    detail::FormatReal(0.1) + " " + detail::FormatReal(1.0 / 3.0);
  const auto text = detail::SerializeDescription(description);
  CHECK(detail::ParseDescription(text) == description);
  CHECK(detail::ParseReal(detail::FormatReal(1.0 / 3.0)) == 1.0 / 3.0);
  CHECK(detail::ParseInteger("-42") == -42);
  CHECK_THROWS(detail::ParseReal("0.1x"));
  CHECK_THROWS(detail::Require(description, "function.1.name"));
}

TEST_CASE("a setup description of another schema or PCMS version is rejected")
{
  auto description = detail::CreateDescription();
  description["pcms_version"] = "0.0.1";
  CHECK_THROWS(
    detail::ParseDescription(detail::SerializeDescription(description)));
  description = detail::CreateDescription();
  description["pcms_schema"] = "0";
  CHECK_THROWS(
    detail::ParseDescription(detail::SerializeDescription(description)));
  CHECK_THROWS(detail::ParseDescription("function.0.name=f\n"));
  CHECK_THROWS(detail::SerializeDescription({{"a=b", "c"}}));
}

TEST_CASE("coordinate systems, value bases and metadata are described and "
          "rebuilt")
{
  detail::Description description;
  detail::DescribedSystems systems;
  // y twist-shifted along z by S(x) = 0.1 + 0.2 x, z periodic
  auto shift = std::make_shared<pcms::Function<Real>>(
    pcms::CreateLagrangeSpace(pcms::uniform_grid::CreateGeometry(
                                pcms::UniformGrid<1>{.edge_length = {1.0},
                                                     .bot_left = {0.0},
                                                     .divisions = {2}},
                                pcms::csys::CreateCartesian(1)),
                              1, 1)
      ->CreateFunction<Real>("S"));
  pcms::test::SetFieldComponents(
    *shift, [](Real x, Real* out) { out[0] = 0.1 + 0.2 * x; });
  const auto parameter = pcms::csys::CreateParameterSystem(
    "bout_index", 3,
    {{1, 0.0, 1.0, pcms::CreateTwistShiftTransform(2, shift, {0})},
     {2, 0.5, 2.0}});
  const pcms::ValueBasis scalar{.units = pcms::Units::Parse("Pa")};
  const pcms::ValueBasis covector{pcms::csys::CreateCylindricalRZ(),
                                  pcms::ComponentScaling::Natural,
                                  pcms::values::Covector, pcms::Parity::Odd};
  const pcms::ValueBasis tensor{
    pcms::csys::CreateCartesian(3),
    pcms::ComponentScaling::Physical,
    {pcms::Variance::Contravariant, pcms::Variance::Covariant},
    pcms::Parity::Even};
  const pcms::ValueBasis metric{parameter, std::nullopt, pcms::values::Tensor};
  const pcms::ValueBasis array{.kind = pcms::ValueKind::Array};
  const std::vector<pcms::ValueBasis> bases{
    scalar,
    covector,
    tensor,
    metric,
    array,
    pcms::ValueBasis::Coordinates(pcms::csys::CreateCylindricalRThetaZ())};
  for (size_t i = 0; i < bases.size(); ++i) {
    detail::DescribeValueBasis(bases[i], "basis." + std::to_string(i), systems,
                               description);
  }
  // the parameter system is described once for its two uses
  detail::DescribeValueBasis(metric, "again", systems, description);
  CHECK(systems.size() == 5);
  const auto rebuilt_systems = detail::SystemsFromDescription(description);
  REQUIRE(rebuilt_systems.size() == systems.size());
  for (size_t i = 0; i < bases.size(); ++i) {
    const auto rebuilt = detail::ValueBasisFromDescription(
      description, "basis." + std::to_string(i), rebuilt_systems);
    CHECK(rebuilt.variance == bases[i].variance);
    CHECK(rebuilt.scaling == bases[i].scaling);
    CHECK(rebuilt.parity == bases[i].parity);
    CHECK(rebuilt.kind == bases[i].kind);
    CHECK(rebuilt.units.has_value() == bases[i].units.has_value());
    if (bases[i].system != parameter) {
      CHECK(detail::SameCoordinateSystem(rebuilt.system, bases[i].system));
    }
  }
  const auto rebuilt_parameter =
    detail::ValueBasisFromDescription(description, "basis.3", rebuilt_systems)
      .system;
  CHECK(rebuilt_parameter ==
        detail::ValueBasisFromDescription(description, "again", rebuilt_systems)
          .system);
  CHECK(rebuilt_parameter->Kind() == "bout_index");
  CHECK(rebuilt_parameter->Dimension() == 3);
  const auto periodic_axes = rebuilt_parameter->PeriodicAxes();
  REQUIRE(periodic_axes.size() == 2);
  CHECK((periodic_axes[1].axis == 2 && periodic_axes[1].lower == 0.5 &&
         periodic_axes[1].length == 2.0 && !periodic_axes[1].transform));
  CHECK((periodic_axes[0].axis == 1 && periodic_axes[0].length == 1.0));
  REQUIRE(periodic_axes[0].transform);
  const auto* twist_shift =
    detail::GetTwistShiftArguments(*periodic_axes[0].transform);
  REQUIRE(twist_shift);
  CHECK((twist_shift->shift_axis == 2 && twist_shift->input_axes.size() == 1 &&
         twist_shift->input_axes[0] == 0));
  const auto rebuilt_shift = twist_shift->shift->GetDOFHolderDataHost();
  const auto original_shift = shift->GetDOFHolderDataHost();
  REQUIRE(rebuilt_shift.GetValues().extent(0) == 3);
  for (size_t h = 0; h < 3; ++h) {
    CHECK(rebuilt_shift.GetValues()(h, 0) == original_shift.GetValues()(h, 0));
  }

  const auto rebuilt_scalar =
    detail::ValueBasisFromDescription(description, "basis.0", rebuilt_systems);
  REQUIRE(rebuilt_scalar.units);
  CHECK(UnitsText(rebuilt_scalar.units) == UnitsText(scalar.units));

  const pcms::FieldMetadata metadata{.name = "P",
                                     .duality = pcms::Duality::Primal};
  detail::DescribeMetadata(metadata, "function.0", description);
  const auto rebuilt =
    detail::MetadataFromDescription(description, "function.0");
  CHECK(rebuilt.name == "P");
  CHECK(rebuilt.duality == pcms::Duality::Primal);
  detail::DescribeMetadata({.name = "Q"}, "function.1", description);
  const auto bare = detail::MetadataFromDescription(description, "function.1");
  CHECK((bare.name == "Q" && !bare.duality));
}

namespace
{

// a 1/5 toroidal wedge of a sheared annulus: (R, phi, Z) of the logical
// coordinates, trilinear
void WedgePosition(Real x, Real y, Real z, Real* out)
{
  out[0] = 3.0 + 0.5 * x + 0.1 * y;
  out[1] = z + 0.3 * x + 0.2 * x * y;
  out[2] = 0.4 * y + 0.1 * x;
}

std::vector<Real> EvaluateAt(const pcms::FunctionSpace& space,
                             const pcms::Function<Real>& function,
                             std::shared_ptr<const pcms::BoundPoints> bound)
{
  Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", bound->NumPoints(),
                                                    1);
  const auto evaluator = space.CreatePointEvaluator<Real>(std::move(bound));
  evaluator->Evaluate(function, pcms::test::TagLike(function, out));
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
  return {host.data(), host.data() + host.extent(0)};
}

std::vector<Real> VolumeFactors(const pcms::BoundPoints& bound)
{
  const auto factors = bound.VolumeFactor();
  Kokkos::View<Real*, pcms::DeviceMemorySpace> copy("volumes",
                                                    factors.extent(0));
  Kokkos::parallel_for(
    "copy",
    Kokkos::RangePolicy<pcms::DeviceMemorySpace::execution_space>(
      0, factors.extent(0)),
    KOKKOS_LAMBDA(const size_t p) { copy(p) = factors(p); });
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), copy);
  return {host.data(), host.data() + host.extent(0)};
}

} // namespace

TEST_CASE("an embedded uniform grid is described and rebuilt")
{
  const Real wedge = 2 * std::numbers::pi / 5;
  pcms::UniformGrid<3> grid{.edge_length = {1.0, 1.0, wedge},
                            .bot_left = {0.0, 0.0, 0.0},
                            .divisions = {4, 5, 6}};
  const auto logical = pcms::csys::CreateParameterSystem("bout_index", 3);
  const auto parameter = pcms::uniform_grid::CreateGeometry(grid, logical);
  auto position = std::make_shared<pcms::Function<Real>>(
    pcms::CreateLagrangeSpace(parameter, 1, 3)
      ->CreateFunction<Real>("position", [wedge] {
        auto basis = pcms::ValueBasis::Coordinates(
          pcms::csys::CreateCylindricalRThetaZ(wedge));
        basis.units = pcms::Units::Parse("m");
        return basis;
      }()));
  pcms::test::SetFieldComponents(*position, WedgePosition);
  auto g_inv = std::make_shared<pcms::Function<Real>>(
    pcms::CreateLagrangeSpace(parameter, 1, 9)
      ->CreateFunction<Real>("g^ij",
                             {.scaling = pcms::ComponentScaling::Natural,
                              .variance = pcms::values::Tensor}));
  pcms::test::SetFieldComponents(*g_inv, [](Real x, Real y, Real, Real* out) {
    std::fill(out, out + 9, 0.0);
    out[0] = 1.0 + x;
    out[4] = 2.0 + y;
    out[8] = 3.0;
  });
  auto volume = std::make_shared<pcms::Function<Real>>(
    pcms::CreateLagrangeSpace(parameter, 1, 1)->CreateFunction<Real>("J"));
  pcms::test::SetFieldComponents(
    *volume, [](Real x, Real, Real, Real* out) { out[0] = 2.0 + x; });
  const auto embedded = pcms::CreateEmbeddedGeometry(
    parameter, position, {.metric = g_inv, .volume = volume});

  detail::DescribedSystems systems;
  auto description = detail::CreateDescription();
  const auto blocks = detail::DescribeEmbedded(*embedded, "geometry.0", systems,
                                               description, MPI_COMM_SELF, 2);
  const auto described = blocks.value_or(std::vector<std::vector<pcms::GO>>{});
  REQUIRE(described.size() == 2);
  description =
    detail::ParseDescription(detail::SerializeDescription(description));
  const auto rebuilt_systems = detail::SystemsFromDescription(description);
  const auto rebuilt = std::dynamic_pointer_cast<const pcms::EmbeddedGeometry>(
    detail::RebuildEmbedded(description, "geometry.0", rebuilt_systems,
                            {described.front()}));
  REQUIRE(rebuilt);
  CHECK(rebuilt->GetParameterSystem()->Kind() == "bout_index");
  REQUIRE(rebuilt->GetMetric());
  CHECK(rebuilt->GetMetric()->GetData().GetValueBasis().system ==
        rebuilt->GetParameterSystem());
  CHECK(UnitsText(rebuilt->GetPosition()->GetData().GetValueBasis().units) ==
        UnitsText(position->GetData().GetValueBasis().units));
  REQUIRE(rebuilt->GetVolume());
  // a geometry linked into a parent cannot be described
  const auto child = pcms::uniform_grid::CreateGeometry(
    grid, pcms::csys::CreateParameterSystem("child", 3));
  const auto chained = pcms::CreateEmbeddedGeometry(
    child,
    pcms::CreateCoordinateMap(pcms::test::MakePositionFunction(
      pcms::CreateLagrangeSpace(child, 1, 3),
      [](const Real* u, Real* x) { std::copy(u, u + 3, x); }, logical)),
    embedded);
  CHECK_THROWS(detail::DescribeEmbedded(*chained, "geometry.1", systems,
                                        description, MPI_COMM_SELF, 1));
  const auto periodic_axes = rebuilt->GetCoordinateSystem()->PeriodicAxes();
  REQUIRE(periodic_axes.size() == 1);
  CHECK((periodic_axes[0].axis == 1 && periodic_axes[0].length == wedge));

  // the same field on both, evaluated at the same physical points, including
  // translates by the periodic axis
  const auto f = KOKKOS_LAMBDA(Real x, Real y, Real z)
  {
    return 1 + x + 2 * y + 3 * z;
  };
  const auto space = pcms::CreateEmbeddedSpace(
    embedded, pcms::CreateLagrangeSpace(parameter, 1, 1));
  const auto rebuilt_space = pcms::CreateEmbeddedSpace(
    rebuilt, pcms::CreateLagrangeSpace(rebuilt->GetParameterGeometry(), 1, 1));
  auto u = space->CreateFunction<Real>("u");
  auto rebuilt_u = rebuilt_space->CreateFunction<Real>("u");
  pcms::test::SetField(u, f);
  pcms::test::SetField(rebuilt_u, f);
  std::vector<Real> points;
  for (const auto& xi : std::vector<std::array<Real, 3>>{
         {0.3, 0.6, 0.5}, {0.7, 0.2, 0.2}, {0.9, 0.9, 1.1}, {0.5, 0.5, 0.05}}) {
    std::array<Real, 3> x{};
    WedgePosition(xi[0], xi[1], xi[2], x.data());
    points.insert(points.end(),
                  {x[0], x[1], x[2], x[0], x[1] + 2 * wedge, x[2]});
  }
  const auto bound = pcms::test::Bind(*embedded, points);
  const auto rebuilt_bound = pcms::test::Bind(*rebuilt, points);
  const auto values = EvaluateAt(*space, u, bound);
  const auto rebuilt_values =
    EvaluateAt(*rebuilt_space, rebuilt_u, rebuilt_bound);
  const auto volumes = VolumeFactors(*bound);
  const auto rebuilt_volumes = VolumeFactors(*rebuilt_bound);
  for (size_t p = 0; p < values.size(); ++p) {
    CAPTURE(p);
    CHECK_THAT(rebuilt_values[p], Catch::Matchers::WithinAbs(values[p], 1e-12));
    CHECK_THAT(rebuilt_volumes[p],
               Catch::Matchers::WithinRel(volumes[p], 1e-12));
  }
}
