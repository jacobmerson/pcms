#include <catch2/catch_test_macros.hpp>

#include "description.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include "pcms/utility/coordinate_systems/cylindrical.hpp"
#include "pcms/utility/detail/coordinate_system.hpp"

namespace detail = pcms::detail;

TEST_CASE("a setup description reads back what was written")
{
  detail::Description description{
    {"pcms_schema", std::string(detail::DescriptionSchema)},
    {"function.0.name", "density = rho"},
    {"geometry.0.bot_left",
     detail::FormatReal(0.1) + " " + detail::FormatReal(1.0 / 3.0)}};
  const auto text = detail::SerializeDescription(description);
  CHECK(detail::ParseDescription(text) == description);
  CHECK(detail::ParseReal(detail::FormatReal(1.0 / 3.0)) == 1.0 / 3.0);
  CHECK(detail::ParseInteger("-42") == -42);
  CHECK_THROWS(detail::ParseReal("0.1x"));
  CHECK_THROWS(detail::Require(description, "function.1.name"));
}

TEST_CASE("a setup description of another schema is rejected")
{
  CHECK_THROWS(detail::ParseDescription("pcms_schema=0\n"));
  CHECK_THROWS(detail::ParseDescription("function.0.name=f\n"));
  CHECK_THROWS(detail::SerializeDescription({{"a=b", "c"}}));
}

TEST_CASE("charts and value bases are described and rebuilt")
{
  detail::Description description;
  for (const auto& chart :
       {pcms::csys::CreateCartesian(2), pcms::csys::CreateCylindricalRZ(),
        pcms::csys::CreateCylindricalRThetaZ()}) {
    detail::DescribeChart(*chart, "chart", description);
    CHECK(*detail::ChartFromDescription(description, "chart") == *chart);
  }
  const pcms::ValueBasis scalar{};
  const pcms::ValueBasis covector{pcms::csys::CreateCylindricalRZ(),
                                  pcms::ComponentScaling::Natural,
                                  pcms::values::Covector, pcms::Parity::Odd};
  const pcms::ValueBasis tensor{
    pcms::csys::CreateCartesian(3),
    pcms::ComponentScaling::Physical,
    {pcms::Variance::Contravariant, pcms::Variance::Covariant},
    pcms::Parity::Even};
  for (const auto& basis : {scalar, covector, tensor}) {
    detail::DescribeValueBasis(basis, "basis", description);
    const auto rebuilt =
      detail::ValueBasisFromDescription(description, "basis");
    CHECK(rebuilt.variance == basis.variance);
    CHECK(rebuilt.scaling == basis.scaling);
    CHECK(rebuilt.parity == basis.parity);
    CHECK(detail::SameCoordinateSystem(rebuilt.system, basis.system));
    description.clear();
  }
}
