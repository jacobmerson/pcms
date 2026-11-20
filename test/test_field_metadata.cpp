#include "periodic_test_utils.hpp"
#include <catch2/matchers/catch_matchers_string.hpp>
#include <pcms/field/field_metadata.hpp>
#include <pcms/geometry/periodic_geometry.hpp>
#include <pcms/transfer/copy.hpp>
#include <pcms/transfer/interpolator.hpp>
#include <pcms/transfer/projection.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>

using Catch::Matchers::ContainsSubstring;
using pcms::FieldMetadata;
using pcms::FieldProperties;
using pcms::FieldProperty;
using pcms::Real;
using pcms::Units;

namespace
{

std::shared_ptr<const pcms::FunctionSpace> Space(int cells)
{
  return pcms::CreatePeriodicLagrangeSpace(pcms::CreatePeriodicGeometry(
    pcms::csys::CreateCartesian(1), cells, 1.0, 0.0));
}

// Every transfer between periodic Lagrange spaces: a copy, an interpolator and
// the projections with the matrix-free solve
struct Transfers
{
  std::shared_ptr<const pcms::FunctionSpace> source_space = Space(8);
  std::shared_ptr<const pcms::FunctionSpace> target_space = Space(16);
  std::shared_ptr<const pcms::FunctionSpace> same_space = Space(8);
  std::vector<
    std::pair<const char*, std::unique_ptr<pcms::TransferOperator<Real>>>>
    operators;
  Transfers()
  {
    operators.emplace_back("copy",
                           pcms::CreateCopy<Real>(source_space, source_space));
    operators.emplace_back("interpolator", pcms::CreateInterpolator<Real>(
                                             source_space, target_space));
    operators.emplace_back("projection",
                           pcms::CreateProjection(source_space, target_space));
    pcms::ProjectionOptions lumped;
    lumped.mass_matrix_type = pcms::MassMatrixType::Lumped;
    operators.emplace_back(
      "lumped projection",
      pcms::CreateProjection(source_space, target_space, lumped));
    pcms::ProjectionOptions control_variate;
    control_variate.control_variate = true;
    operators.emplace_back(
      "control variate projection",
      pcms::CreateProjection(source_space, target_space, control_variate));
  }
  // the space of the target of operator `index`
  const pcms::FunctionSpace& TargetOf(std::size_t index) const
  {
    return index == 0 ? *source_space : *target_space;
  }
};

} // namespace

TEST_CASE("fields carry metadata", "[metadata]")
{
  const auto space = Space(8);
  auto field = space->CreateFunction<Real>("b");

  REQUIRE(field.GetName() == "b");
  REQUIRE(field.GetMetadata().name == "b");
  REQUIRE_FALSE(field.GetMetadata().units.has_value());
  REQUIRE(field.GetMetadata().properties == FieldProperties{});

  field.SetUnits(Units::Parse("T"));
  field.SetProperties({FieldProperty::DivergenceFree});
  REQUIRE(field.GetMetadata().units == Units::Parse("T"));
  REQUIRE(field.GetMetadata().properties.Has(FieldProperty::DivergenceFree));

  field.SetUnits(std::nullopt);
  REQUIRE_FALSE(field.GetMetadata().units.has_value());
}

TEST_CASE("creating a function sets its metadata, which travels with it",
          "[metadata]")
{
  const auto space = Space(8);
  auto field = space->CreateFunction<Real>(FieldMetadata{
    .name = "n",
    .units = Units::Parse("m^-3"),
    .properties = {FieldProperty::Conserved, FieldProperty::NonNegative}});
  REQUIRE(field.GetName() == "n");
  REQUIRE(field.GetMetadata().units == Units::Parse("m^-3"));
  REQUIRE(field.GetMetadata().properties.Has(FieldProperty::Conserved));
  REQUIRE(field.GetMetadata().properties.Has(FieldProperty::NonNegative));
  REQUIRE_FALSE(
    field.GetMetadata().properties.Has(FieldProperty::DivergenceFree));

  const auto moved = std::move(field);
  REQUIRE(moved.GetName() == "n");
  REQUIRE(moved.GetMetadata().units == Units::Parse("m^-3"));
  REQUIRE(moved.GetMetadata().properties.Has(FieldProperty::Conserved));
}

TEST_CASE("transfers check units", "[metadata]")
{
  Transfers transfers;
  for (std::size_t i = 0; i < transfers.operators.size(); ++i) {
    const auto& [name, op] = transfers.operators[i];
    CAPTURE(name);
    auto source = transfers.source_space->CreateFunction<Real>("source");
    auto target = transfers.TargetOf(i).CreateFunction<Real>("target");
    pcms::test::SetDofs(source, std::vector<Real>(8, 1.0));

    SECTION("unknown units are not checked")
    {
      REQUIRE_NOTHROW(op->Apply(source, target));
      source.SetUnits(Units::Parse("T"));
      REQUIRE_NOTHROW(op->Apply(source, target));
      source.SetUnits(std::nullopt);
      target.SetUnits(Units::Parse("T"));
      REQUIRE_NOTHROW(op->Apply(source, target));
    }
    SECTION("the same units pass")
    {
      source.SetUnits(Units::Parse("T"));
      target.SetUnits(Units::Parse("kg/(A*s^2)"));
      REQUIRE_NOTHROW(op->Apply(source, target));
    }
    SECTION("different dimensions throw")
    {
      source.SetUnits(Units::Parse("T"));
      target.SetUnits(Units::Parse("m"));
      REQUIRE_THROWS_WITH(op->Apply(source, target),
                          ContainsSubstring("transfer from 'source' [T] to "
                                            "'target' [m]") &&
                            ContainsSubstring("different dimensions"));
    }
    SECTION("a different scale throws, and nothing is converted")
    {
      source.SetUnits(Units::Parse("gauss"));
      target.SetUnits(Units::Parse("T"));
      REQUIRE_THROWS_WITH(
        op->Apply(source, target),
        ContainsSubstring("differ in scale or offset") &&
          ContainsSubstring("PCMS does not convert units, so convert one of "
                            "the fields"));
    }
  }
}

TEST_CASE("ApplyTranspose checks units", "[metadata]")
{
  Transfers transfers;
  // the interpolator and the projections of the periodic Lagrange spaces
  // support the transpose; the copy does not
  for (std::size_t i = 1; i < 4; ++i) {
    const auto& [name, op] = transfers.operators[i];
    CAPTURE(name);
    REQUIRE(op->SupportsTranspose());
    auto loads = transfers.target_space->CreateFunction<Real>("loads");
    auto functional =
      transfers.source_space->CreateFunction<Real>("functional");
    pcms::test::SetDofs(loads, std::vector<Real>(16, 1.0));

    REQUIRE_NOTHROW(op->ApplyTranspose(loads, functional));
    loads.SetUnits(Units::Parse("N"));
    REQUIRE_NOTHROW(op->ApplyTranspose(loads, functional));
    functional.SetUnits(Units::Parse("kg*m/s^2"));
    REQUIRE_NOTHROW(op->ApplyTranspose(loads, functional));
    functional.SetUnits(Units::Parse("m"));
    REQUIRE_THROWS_WITH(op->ApplyTranspose(loads, functional),
                        ContainsSubstring("transposed transfer from 'loads' "
                                          "[N] to 'functional' [m]"));
    functional.SetUnits(Units::Parse("dyn"));
    REQUIRE_THROWS_WITH(op->ApplyTranspose(loads, functional),
                        ContainsSubstring("differ in scale or offset"));
  }
}

TEST_CASE("a transfer without a transpose still reports it", "[metadata]")
{
  Transfers transfers;
  const auto& copy = transfers.operators[0].second;
  REQUIRE_FALSE(copy->SupportsTranspose());
  auto a = transfers.source_space->CreateFunction<Real>("a");
  auto b = transfers.source_space->CreateFunction<Real>("b");
  REQUIRE_THROWS_WITH(copy->ApplyTranspose(a, b),
                      ContainsSubstring("not supported"));
}
