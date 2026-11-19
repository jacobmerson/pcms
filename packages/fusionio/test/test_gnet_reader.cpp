#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "gnet_reader.hpp"
#include "gnet_synthetic.hpp"
#include <hdf5.h>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <string>

using Catch::Approx;
using namespace fusionio::detail;
namespace test = fusionio::test;

namespace
{

std::string TempFile(const char* name)
{
  return (std::filesystem::temp_directory_path() / name).string();
}

} // namespace

TEST_CASE("GNET reader: dimensions, ranges and volumes", "[gnet]")
{
  const auto path = TempFile("fusionio_test_gnet.h5");
  const test::GnetShape shape{4, 3, 5, 4};
  test::WriteSyntheticGnet(path, shape);
  const GnetData data = ReadGnet(path);

  CHECK(data.num_rho == 4);
  CHECK(data.num_zeta == 3);
  CHECK(data.num_energy == 5);
  CHECK(data.num_pitch == 4);
  CHECK(data.max_energy_ev == test::SyntheticMaxEnergyEv);
  CHECK(data.max_zeta_rad == test::SyntheticMaxZetaRad);

  // 4 + 8 + 12 + 16 poloidal bins
  CHECK(GnetData::NumPoloidalBins(2) == 12);
  CHECK(GnetData::PoloidalOffset(0) == 0);
  CHECK(GnetData::PoloidalOffset(3) == 24);
  CHECK(GnetData::PoloidalOffset(4) == 40);
  CHECK(data.NumCells() == 120);
  CHECK(data.NumMarkers() == 2400);
  REQUIRE(data.volume3d.size() == 120);
  REQUIRE(data.f5d.size() == 2400);

  // the cell of (rho 2, theta 5, zeta 1)
  const std::size_t cell = (GnetData::PoloidalOffset(2) + 5) * 3 + 1;
  const double rho = 2.5 / 4;
  const double theta = 5.5 * 2.0 * std::numbers::pi / 12;
  CHECK(data.volume3d[cell] == Approx(test::SyntheticVolume(rho, theta)));
  const double zeta = 1.5 * test::SyntheticMaxZetaRad / 3;
  const double energy = test::SyntheticMaxEnergyEv * 3.5 / 5;
  const double pitch = 2.0 * 1.5 / 4 - 1.0;
  const std::size_t marker = (cell * 5 + 3) * 4 + 1;
  CHECK(data.f5d[marker] ==
        Approx(data.volume3d[cell] *
               test::SyntheticNumberDensity(rho, theta, zeta, energy, pitch)));

  const auto density = data.NumberDensity();
  REQUIRE(density.size() == data.NumMarkers());
  CHECK(density[marker] ==
        Approx(test::SyntheticNumberDensity(rho, theta, zeta, energy, pitch)));
  for (const double n : density) {
    REQUIRE(n > 0);
  }
  std::filesystem::remove(path);
}

TEST_CASE("GNET reader: errors", "[gnet]")
{
  H5Eset_auto(H5E_DEFAULT, nullptr, nullptr); // the throw reports the error
  CHECK_THROWS_WITH(ReadGnet(TempFile("fusionio_no_such_gnet.h5")),
                    Catch::Matchers::ContainsSubstring("HDF5 file"));
}
