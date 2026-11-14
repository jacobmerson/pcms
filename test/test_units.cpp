#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "pcms/field/field_metadata.hpp"
#include "pcms/field/units.hpp"
#include "pcms/utility/assert.h"

using Catch::Matchers::ContainsSubstring;
using pcms::FieldMetadata;
using pcms::FieldProperties;
using pcms::FieldProperty;
using pcms::Units;
using pcms::UnitSyntax;

TEST_CASE("Units parse SI, IMAS and fusion notations")
{
  REQUIRE(Units::Parse("T") == Units::Parse("kg/(A*s^2)"));
  REQUIRE(Units::Parse("Tesla") == Units::Parse("T"));
  REQUIRE(Units::Parse("m/s") == Units::Parse("m.s^-1"));
  REQUIRE(Units::Parse("A.m^-2") == Units::Parse("A/m^2"));
  REQUIRE(Units::Parse("m^-3") == Units::Parse("1/m^3"));
  REQUIRE(Units::Parse("  eV ").SameDimensions(Units::Parse("J")));
  REQUIRE_FALSE(Units::Parse("eV") == Units::Parse("J"));
}

TEST_CASE("Units compare dimensions, scale and offset")
{
  const auto tesla = Units::Parse("T");
  const auto gauss = Units::Parse("gauss");
  REQUIRE(tesla.SameDimensions(gauss));
  REQUIRE_FALSE(tesla == gauss);
  REQUIRE_FALSE(tesla.SameDimensions(Units::Parse("m")));
  REQUIRE(Units::Parse("degC").SameDimensions(Units::Parse("K")));
  REQUIRE_FALSE(Units::Parse("degC") == Units::Parse("K"));
}

TEST_CASE("Units round-trip through ToString")
{
  for (const char* text : {"T", "gauss", "m/s", "A/m^2", "eV", "1/m^3", "Pa"}) {
    const auto units = Units::Parse(text);
    REQUIRE(Units::Parse(units.ToString()) == units);
  }
}

TEST_CASE("Units parse CF strings")
{
  REQUIRE(Units::Parse("m2 s-1", UnitSyntax::CF) == Units::Parse("m^2/s"));
  REQUIRE(Units::Parse("J kg-1 K-1", UnitSyntax::CF) ==
          Units::Parse("J/(kg*K)"));
  REQUIRE(Units::Parse("kg m-2 s-1", UnitSyntax::CF) ==
          Units::Parse("kg/(m^2*s)"));
  // Without the CF syntax LLNL/units reads this as 1/mol^2.
  REQUIRE(Units::Parse("mol mol-1", UnitSyntax::CF) == Units::Parse("1"));
  REQUIRE(Units::Parse("degrees_Celsius", UnitSyntax::CF) ==
          Units::Parse("degC"));
  REQUIRE_THROWS_AS(Units::Parse("days since 2000-01-01", UnitSyntax::CF),
                    pcms::pcms_error);
}

TEST_CASE("Units parse IMAS strings")
{
  REQUIRE(Units::Parse("-", UnitSyntax::IMAS) == Units::Parse("1"));
  REQUIRE(Units::Parse("W.m^-3", UnitSyntax::IMAS) == Units::Parse("W/m^3"));
  REQUIRE_THROWS_WITH(Units::Parse("mixed", UnitSyntax::IMAS),
                      ContainsSubstring("leave the units unspecified"));
  REQUIRE_THROWS_AS(Units::Parse("as_parent", UnitSyntax::IMAS),
                    pcms::pcms_error);
}

TEST_CASE("Units reject what is not a unit")
{
  REQUIRE_THROWS_AS(Units::Parse(""), pcms::pcms_error);
  REQUIRE_THROWS_AS(Units::Parse("  "), pcms::pcms_error);
  REQUIRE_THROWS_AS(Units::Parse("-"), pcms::pcms_error);
  REQUIRE_THROWS_WITH(Units::Parse("UNITLESS"),
                      ContainsSubstring("\"UNITLESS\" is not a unit"));
}

TEST_CASE("FieldProperties hold a set of properties")
{
  const FieldProperties none;
  REQUIRE_FALSE(none.Has(FieldProperty::Conserved));
  const FieldProperties b{FieldProperty::DivergenceFree};
  REQUIRE(b.Has(FieldProperty::DivergenceFree));
  REQUIRE_FALSE(b.Has(FieldProperty::NonNegative));
  const FieldProperties n{FieldProperty::Conserved, FieldProperty::NonNegative};
  REQUIRE(n.Has(FieldProperty::Conserved));
  REQUIRE(n.Has(FieldProperty::NonNegative));
  REQUIRE(
    n == FieldProperties{FieldProperty::NonNegative, FieldProperty::Conserved});
}

TEST_CASE("CheckSameUnits")
{
  const FieldMetadata tesla{.name = "b", .units = Units::Parse("T")};
  const FieldMetadata gauss{.name = "b_g", .units = Units::Parse("gauss")};
  const FieldMetadata metre{.name = "x", .units = Units::Parse("m")};
  const FieldMetadata unknown{.name = "u"};

  SECTION("passes for the same or unknown units")
  {
    REQUIRE_NOTHROW(CheckSameUnits(tesla, tesla, "transfer"));
    REQUIRE_NOTHROW(CheckSameUnits(
      tesla, FieldMetadata{.name = "c", .units = Units::Parse("kg/(A*s^2)")},
      "transfer"));
    REQUIRE_NOTHROW(CheckSameUnits(tesla, unknown, "transfer"));
    REQUIRE_NOTHROW(CheckSameUnits(unknown, tesla, "transfer"));
  }
  SECTION("throws for different dimensions")
  {
    REQUIRE_THROWS_WITH(CheckSameUnits(tesla, metre, "transfer"),
                        ContainsSubstring("transfer from 'b' [T] to 'x' [m]") &&
                          ContainsSubstring("different dimensions"));
  }
  SECTION("throws for a different scale")
  {
    REQUIRE_THROWS_WITH(CheckSameUnits(gauss, tesla, "transfer"),
                        ContainsSubstring("differ in scale or offset"));
  }
}
