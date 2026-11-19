#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "fusionio/gnet.hpp"
#include "gnet_synthetic.hpp"
#include <hdf5.h>
#include <pcms/field/query_points.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using fusionio::MarkerRows;
using pcms::Real;
namespace test = fusionio::test;

namespace
{

std::string TempFile(const char* name)
{
  return (std::filesystem::temp_directory_path() / name).string();
}

std::vector<Real> Values(const pcms::Function<Real>& f,
                         const std::vector<std::array<Real, 5>>& points)
{
  const std::size_t n = points.size();
  Kokkos::View<Real**, pcms::HostMemorySpace> host("points", n, 5);
  for (std::size_t i = 0; i < n; ++i) {
    for (int d = 0; d < 5; ++d) {
      host(i, d) = points[i][d];
    }
  }
  Kokkos::View<Real**, pcms::DeviceMemorySpace> device("points", n, 5);
  pcms::DeepCopyMismatchLayouts(device, host);
  Kokkos::View<Real**, pcms::DeviceMemorySpace> out("values", n, 1);
  pcms::Evaluate(
    f,
    pcms::CoordinateView<pcms::DeviceMemorySpace>(
      f.GetSpace().GetCoordinateSystem(), pcms::MakeConstRank2View(device)),
    pcms::ValueView<Real, pcms::DeviceMemorySpace>(f.GetData().GetValueBasis(),
                                                   pcms::MakeRank2View(out)),
    pcms::DifferentialOperator::Value,
    pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, -1.0});
  const auto host_out =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
  std::vector<Real> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = host_out(i, 0);
  }
  return values;
}

} // namespace

TEST_CASE("GNET distribution: the reconstruction of the synthetic file",
          "[gnet]")
{
  const auto path = TempFile("fusionio_test_gnet_density.h5");
  test::WriteSyntheticGnet(path);
  for (const bool periodic : {false, true}) {
    const auto gnet = fusionio::OpenGnet(path, {.periodic = periodic});
    CHECK(gnet.max_energy == test::SyntheticMaxEnergyEv);
    CHECK(gnet.max_zeta == test::SyntheticMaxZetaRad);
    CHECK(gnet.min_volume > 0);
    CHECK(gnet.max_volume > gnet.min_volume);
    REQUIRE(gnet.density.GetSpace().GetCoordinateSystem()->Dimension() == 5);
    // the density is smooth: the reconstruction of degree 1 is close to it
    // inside the grid, away from the seams
    std::vector<std::array<Real, 5>> points;
    for (const Real rho : {0.3, 0.6}) {
      for (const Real theta : {1.0, 2.5}) {
        for (const Real energy : {30000.0, 100000.0}) {
          points.push_back({rho, theta, 0.5, energy, 0.1});
        }
      }
    }
    const auto values = Values(gnet.density, points);
    for (std::size_t i = 0; i < points.size(); ++i) {
      const auto& p = points[i];
      const Real exact =
        test::SyntheticNumberDensity(p[0], p[1], p[2], p[3], p[4]);
      CHECK(std::abs(values[i] - exact) < 0.25 * exact);
    }
  }
  std::filesystem::remove(path);
}

TEST_CASE("GNET distribution: options and errors", "[gnet]")
{
  H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
  CHECK_THROWS_WITH(fusionio::OpenGnet(TempFile("fusionio_no_such_gnet.h5")),
                    ContainsSubstring("HDF5 file"));
  const auto path = TempFile("fusionio_test_gnet_options.h5");
  test::WriteSyntheticGnet(path);
  CHECK_THROWS_WITH(fusionio::OpenGnet(path, {.degree = 0}),
                    ContainsSubstring("degree"));
  // degree 3 needs three bins of each coordinate around a point and more
  // supports than the search provides at degree 5
  CHECK_THROWS(fusionio::OpenGnet(path, {.degree = 6}));
  std::filesystem::remove(path);
}

TEST_CASE("SampleMarkers: GNET markers of the synthetic file", "[gnet]")
{
  const auto path = TempFile("fusionio_test_gnet_markers.h5");
  test::WriteSyntheticGnet(path);
  const auto gnet = fusionio::OpenGnet(path);
  const Real threshold = 1.0 / gnet.max_volume;
  const auto domain = fusionio::GTCDomain();
  const fusionio::SamplingOptions options{
    .samples = 3000, .batch = 1000, .seed = 14000, .threshold = threshold};
  const auto markers = fusionio::SampleMarkers(gnet.density, domain, options);
  REQUIRE(markers.rows.extent(0) == 3000);
  REQUIRE(markers.rows.extent(1) == 6);
  const auto rows =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), markers.rows);
  for (std::size_t i = 0; i < rows.extent(0); ++i) {
    for (int d = 0; d < 5; ++d) {
      REQUIRE(rows(i, d) >= domain[d].lo);
      REQUIRE(rows(i, d) <= domain[d].hi);
    }
    REQUIRE(rows(i, 5) >= threshold);
  }
  // the threshold keeps about half of the candidates of the synthetic file
  const double acceptance =
    static_cast<double>(markers.rows.extent(0)) / markers.candidates;
  CHECK(acceptance > 0.3);
  CHECK(acceptance < 0.7);

  // the file of /gtc_n6d: written in chunks, read back
  const auto out = TempFile("fusionio_test_gtc.h5");
  {
    fusionio::GTCMarkerWriter writer(out, options.samples);
    fusionio::SampleMarkers(
      gnet.density, domain, options,
      [&](const MarkerRows& chunk) { writer.Append(chunk); });
    CHECK(writer.Written() == 3000);
    CHECK_THROWS_WITH(writer.Append(markers.rows),
                      ContainsSubstring("more markers"));
  }
  const hid_t file = H5Fopen(out.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(file >= 0);
  const hid_t dataset = H5Dopen(file, "/gtc_n6d", H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  const hid_t space = H5Dget_space(dataset);
  hsize_t dims[2] = {};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  CHECK(dims[0] == 3000);
  CHECK(dims[1] == 6);
  std::vector<double> table(3000 * 6);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          table.data());
  H5Sclose(space);
  H5Dclose(dataset);
  H5Fclose(file);
  for (std::size_t i = 0; i < 3000; ++i) {
    for (int c = 0; c < 6; ++c) {
      REQUIRE(table[6 * i + c] == rows(i, c));
    }
  }
  std::filesystem::remove(path);
  std::filesystem::remove(out);
}
