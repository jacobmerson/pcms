#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include "pcms/field/coordinate_systems/cartesian.hpp"
#include "pcms/field/function_space/lagrange.h"
#include "pcms/transfer/copy.h"
#include "pcms/transfer/interpolator.h"

using Catch::Matchers::ContainsSubstring;
using pcms::FieldProperties;
using pcms::FieldProperty;
using pcms::Real;
using pcms::Units;

namespace
{

Omega_h::Mesh Mesh(Omega_h::Library& lib)
{
  return Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 4, 4, 0,
                            false);
}

// The space refers to `mesh`, which must outlive it.
std::shared_ptr<pcms::LagrangeFunctionSpace> Space(Omega_h::Mesh& mesh)
{
  return pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, pcms::csys::Cartesian::Deferred(), "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
}

} // namespace

TEST_CASE("fields carry metadata")
{
  auto lib = Omega_h::Library{};
  auto mesh = Mesh(lib);
  const auto space = Space(mesh);
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

TEST_CASE("transfers check units")
{
  auto lib = Omega_h::Library{};
  auto mesh = Mesh(lib);
  const auto space = Space(mesh);
  auto source = space->CreateFunction<Real>("source");
  auto target = space->CreateFunction<Real>("target");
  const pcms::Copy<Real> copy(*space, *space);
  const pcms::Interpolator<Real> interpolate(*space, *space);

  SECTION("unknown units are not checked")
  {
    source.SetUnits(Units::Parse("T"));
    REQUIRE_NOTHROW(copy.Apply(source, target));
    REQUIRE_NOTHROW(interpolate.Apply(source, target));
  }
  SECTION("the same units pass")
  {
    source.SetUnits(Units::Parse("T"));
    target.SetUnits(Units::Parse("kg/(A*s^2)"));
    REQUIRE_NOTHROW(copy.Apply(source, target));
    REQUIRE_NOTHROW(interpolate.Apply(source, target));
  }
  SECTION("different dimensions throw")
  {
    source.SetUnits(Units::Parse("T"));
    target.SetUnits(Units::Parse("m"));
    REQUIRE_THROWS_WITH(copy.Apply(source, target),
                        ContainsSubstring("transfer from 'source' [T] to "
                                          "'target' [m]"));
    REQUIRE_THROWS_WITH(interpolate.Apply(source, target),
                        ContainsSubstring("different dimensions"));
  }
  SECTION("a different scale throws")
  {
    source.SetUnits(Units::Parse("gauss"));
    target.SetUnits(Units::Parse("T"));
    REQUIRE_THROWS_WITH(copy.Apply(source, target),
                        ContainsSubstring("differ in scale or offset"));
  }
}
