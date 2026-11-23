#include "bout_test_utils.hpp"
#include <pcms/field/function_spaces.hpp>
#include <pcms/transfer/l2_inner_product.hpp>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::Real;

namespace
{

// BOUT.inp of BOUT++'s elm-pb example: MZ = 16, zperiod = 15
constexpr int Nz = 16;
constexpr int Zperiod = 15;
constexpr Real Period = 2 * std::numbers::pi_v<Real> / Zperiod;

const std::string Grid = FUSIONIO_TEST_BOUT_GRID;

pcms::Function<Real> Ones(const pcms::FunctionSpace& space)
{
  auto one = space.CreateFunction<Real>("one");
  Kokkos::View<Real**, pcms::HostMemorySpace> values(
    "one", space.GetLayout()->GetNumDofHolder(), 1);
  Kokkos::deep_copy(values, 1.0);
  one.SetDOFHolderDataHost(pcms::ValueView<const Real, pcms::HostMemorySpace>(
    one.GetData().GetValueBasis(), pcms::MakeConstRank2View(values)));
  return one;
}

} // namespace

TEST_CASE("BOUT++ CBM18: the grid file read and built", "[bout][cbm18]")
{
  const auto file = fusionio::ReadBoutGrid(Grid, Nz, Zperiod);
  const auto& grid = file.grid;
  const auto& a = file.arrays;
  REQUIRE(grid.nx == 68);
  REQUIRE(grid.ny == 64);
  const auto spaces = fusionio::Build(grid);
  const auto source = fusionio::OpenBout(Grid, Nz, Zperiod);
  const auto at = [&](const char* name, int x, int y) {
    return a.at(name)[x * grid.ny + y];
  };

  SECTION("the position at the points is (Rxy, z dz + zShift, Zxy), past "
          "the seam shifted by ShiftAngle")
  {
    const auto pos = fusionio::test::AtIndex(*spaces.geometry->GetPosition(),
                                             {2, 0, 0, 30, 17, 5, 65, 63, 15});
    const int points[3][3] = {{2, 0, 0}, {30, 17, 5}, {65, 63, 15}};
    for (int p = 0; p < 3; ++p) {
      const auto [x, y, z] = points[p];
      CAPTURE(p);
      CHECK_THAT(pos(p, 0), WithinAbs(at("Rxy", x, y), 1e-13));
      CHECK_THAT(pos(p, 1), WithinAbs(z * grid.dz + at("zShift", x, y), 1e-13));
      CHECK_THAT(pos(p, 2), WithinAbs(at("Zxy", x, y), 1e-13));
    }
    for (const Real x : {2.0, 20.5, 47.25, 65.0}) {
      for (const Real z : {0.0, 7.5, 15.9}) {
        fusionio::test::CheckSeamPosition(spaces, 64, x, z, Period, 1e-5);
      }
    }
  }

  SECTION("zShift follows the field line: dzShift/dy = hthe Bt / (R Bp), "
          "once around it ShiftAngle")
  {
    for (const int x : {2, 33, 65}) {
      CAPTURE(x);
      const auto nu = [&](int y) {
        return at("hthe", x, y) * at("Btxy", x, y) /
               (at("Rxy", x, y) * at("Bpxy", x, y));
      };
      for (const int y : {1, 20, 62}) {
        CAPTURE(y);
        const Real slope = (at("zShift", x, y + 1) - at("zShift", x, y - 1)) /
                           (2 * at("dy", x, y));
        CHECK_THAT(slope, WithinRel(nu(y), 1e-2));
      }
      // the trapezoid rule over the last interval
      const Real end =
        at("zShift", x, 63) + 0.5 * at("dy", x, 63) * (nu(63) + nu(0));
      CHECK_THAT(end, WithinAbs(at("zShift", x, 0) + a.at("ShiftAngle")[x],
                                1e-3 * std::abs(a.at("ShiftAngle")[x])));
    }
  }

  SECTION("the volume is the sum of J dx dy dz, J = hthe / Bp")
  {
    // x by the trapezoid rule over the interior points
    Real sum = 0;
    for (int x = grid.mxg; x < grid.nx - grid.mxg; ++x) {
      const Real weight =
        (x == grid.mxg || x == grid.nx - grid.mxg - 1) ? 0.5 : 1.0;
      for (int y = 0; y < grid.ny; ++y) {
        sum += weight * at("hthe", x, y) / at("Bpxy", x, y) * at("dx", x, y) *
               at("dy", x, y);
      }
    }
    sum *= Nz * grid.dz;
    // the metric induced by the position
    const auto induced = pcms::CreateEmbeddedSpace(
      pcms::CreateEmbeddedGeometry(spaces.geometry->GetParameterGeometry(),
                                   spaces.geometry->GetPosition()),
      pcms::CreateLagrangeSpace(spaces.geometry->GetParameterGeometry(), 1, 1));
    CHECK_THAT(pcms::L2Integral(induced).Apply(Ones(*induced)),
               WithinRel(sum, 2e-2));
    // the multilinear g^ij of the sheared metric loses 16% of the volume
    const Real volume =
      pcms::L2Integral(spaces.space).Apply(Ones(*spaces.space));
    CHECK_THAT(volume, WithinRel(0.844 * sum, 1e-2));
  }

  SECTION("the source has the geometry of Build and the file's profiles")
  {
    const auto& geometry =
      dynamic_cast<const pcms::EmbeddedGeometry&>(*source->GetGeometry());
    const std::vector<Real> points = {3.5, 10.25, 2.5, 40, 63.5, 11};
    const auto expected =
      fusionio::test::AtIndex(*spaces.geometry->GetPosition(), points);
    const auto pos = fusionio::test::AtIndex(*geometry.GetPosition(), points);
    for (int p = 0; p < 2; ++p) {
      for (int c = 0; c < 3; ++c) {
        CHECK(pos(p, c) == expected(p, c));
      }
    }
    CHECK(source->Quantities().size() == 3);
    CHECK_THAT(source->ToroidalPeriod(), WithinRel(Period, 1e-15));
    const auto pressure =
      source->CreateFunction(fusionio::Quantity::TotalPressure);
    REQUIRE(pressure.GetMetadata().units.has_value());
    CHECK(pressure.GetMetadata().units->ToString() == "Pa");
    const auto electrons =
      source->CreateFunction(fusionio::Quantity::Temperature,
                             {.species = fusionio::Species::Electron});
    const auto p = fusionio::test::AtIndex(pressure, {12, 5, 3, 12, 64, 3});
    CHECK_THAT(p(0, 0), WithinRel(at("pressure", 12, 5), 1e-14));
    CHECK_THAT(p(1, 0), WithinRel(at("pressure", 12, 0), 1e-14));
    const auto te = fusionio::test::AtIndex(electrons, {12, 5, 3});
    CHECK_THAT(te(0, 0), WithinRel(at("Te0", 12, 5), 1e-14));
    CHECK_THROWS(source->CreateFunction(
      fusionio::Quantity::Density, {.species = fusionio::Species::Electron}));
    CHECK_THROWS(source->CreateFunction(fusionio::Quantity::MagneticField));
  }
}
