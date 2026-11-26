#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "fusionio/sampling.hpp"
#include <pcms/backends/point_cloud/geometry.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/utility/coordinate_systems/parameter.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using fusionio::Domain;
using fusionio::Draw;
using fusionio::MarkerRows;
using fusionio::SamplingOptions;
using pcms::Real;

namespace
{

// the linear function 1 + x + y of 2D random sites on [-0.1, 1.1]^2 (degree-1
// moving least squares reproduces it)
pcms::Function<Real> LinearFunction()
{
  constexpr int NumSites = 400;
  std::mt19937 gen(3);
  std::uniform_real_distribution<Real> dist(-0.1, 1.1);
  std::vector<Real> sites(2 * NumSites);
  for (auto& s : sites) {
    s = dist(gen);
  }
  auto space = pcms::CreatePolynomialReconstructionSpace(
    pcms::point_cloud::CreateGeometry(
      pcms::CoordinateView<pcms::HostMemorySpace>(
        pcms::csys::CreateParameterSystem("xy", 2),
        pcms::Rank2View<Real, pcms::HostMemorySpace>(sites.data(), NumSites,
                                                     2)),
      pcms::NeighbourhoodOptions{.radius = 0.5, .min_supports = 10}),
    pcms::MLSOptions{.degree = 1, .tol = 1e-12});
  auto f = space->CreateFunction<Real>();
  std::vector<Real> values(NumSites);
  for (int i = 0; i < NumSites; ++i) {
    values[i] = 1 + sites[2 * i] + sites[2 * i + 1];
  }
  f.SetDOFHolderDataHost(pcms::ValueView<const Real, pcms::HostMemorySpace>(
    pcms::ValueBasis{}, pcms::Rank2View<const Real, pcms::HostMemorySpace>(
                          values.data(), NumSites, 1)));
  return f;
}

Kokkos::View<Real**, Kokkos::LayoutRight, pcms::HostMemorySpace> ToHost(
  const MarkerRows& rows)
{
  return Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), rows);
}

// the acceptance of the full rounds of a streamed run
double Acceptance(const pcms::Function<Real>& f, const Domain& domain,
                  const SamplingOptions& options)
{
  std::vector<std::size_t> takes;
  fusionio::SampleMarkers(f, domain, options, [&](const MarkerRows& rows) {
    takes.push_back(rows.extent(0));
  });
  REQUIRE(takes.size() > 3);
  takes.pop_back(); // the last round is clipped
  std::size_t accepted = 0;
  for (const auto take : takes) {
    accepted += take;
  }
  return static_cast<double>(accepted) / (takes.size() * options.batch);
}

} // namespace

TEST_CASE("SampleMarkers: the markers of a linear function", "[sampling]")
{
  const auto f = LinearFunction();
  const Domain domain{{0, 1}, {0, 1}};
  // x + y >= 1 holds on half of the square
  const SamplingOptions options{
    .samples = 5000, .batch = 1000, .seed = 5, .threshold = 2.0};
  const auto markers = fusionio::SampleMarkers(f, domain, options);
  REQUIRE(markers.rows.extent(0) == 5000);
  REQUIRE(markers.rows.extent(1) == 3);
  const auto rows = ToHost(markers.rows);
  for (std::size_t i = 0; i < rows.extent(0); ++i) {
    REQUIRE(rows(i, 0) >= 0);
    REQUIRE(rows(i, 0) <= 1);
    REQUIRE(rows(i, 1) >= 0);
    REQUIRE(rows(i, 1) <= 1);
    REQUIRE(rows(i, 2) >= 2.0);
    REQUIRE_THAT(rows(i, 2), WithinAbs(1 + rows(i, 0) + rows(i, 1), 1e-8));
  }
  // 5000 of about half of the candidates, in rounds of 1000
  REQUIRE(markers.candidates % 1000 == 0);
  REQUIRE(markers.candidates >= 9000);
  REQUIRE(markers.candidates <= 11000);

  // the acceptance of the full rounds: 0.5 within 5 standard deviations
  // (sigma 0.5 / sqrt(20000) = 0.0035)
  REQUIRE_THAT(
    Acceptance(f, domain,
               {.samples = 10000, .batch = 2000, .seed = 6, .threshold = 2.0}),
    WithinAbs(0.5, 0.018));
}

TEST_CASE("SampleMarkers: a sqrt axis is uniform in area", "[sampling]")
{
  const auto f = LinearFunction();
  // on the line y = 0 the function is 1 + x, which is >= 1.5 for x >= 0.5:
  // P(sqrt(u) >= 0.5) = 3/4 for the sqrt draw of x, 1/2 for a uniform one
  const double threshold = 1.5;
  const double sqrt_draw = Acceptance(
    f, {{0, 1, Draw::Sqrt}, {0, 0}},
    {.samples = 15000, .batch = 2000, .seed = 7, .threshold = threshold});
  const double uniform = Acceptance(
    f, {{0, 1}, {0, 0}},
    {.samples = 10000, .batch = 2000, .seed = 7, .threshold = threshold});
  // 5 standard deviations of the 7 or so rounds
  REQUIRE_THAT(sqrt_draw, WithinAbs(0.75, 0.025));
  REQUIRE_THAT(uniform, WithinAbs(0.5, 0.03));
}

TEST_CASE("SampleMarkers: streaming gives the same markers", "[sampling]")
{
  const auto f = LinearFunction();
  const Domain domain{{0, 1, Draw::Sqrt}, {0, 1}};
  // 7001 is not a multiple of the accepted numbers of the rounds: the last is
  // clipped
  const SamplingOptions options{
    .samples = 7001, .batch = 1000, .seed = 9, .threshold = 1.5};
  const auto markers = fusionio::SampleMarkers(f, domain, options);
  REQUIRE(markers.rows.extent(0) == 7001);
  const auto expected = ToHost(markers.rows);
  std::size_t next = 0;
  std::size_t rounds = 0;
  const auto candidates =
    fusionio::SampleMarkers(f, domain, options, [&](const MarkerRows& rows) {
      REQUIRE(rows.extent(1) == 3);
      REQUIRE(rows.extent(0) <= 1000);
      const auto chunk = ToHost(rows);
      for (std::size_t i = 0; i < chunk.extent(0); ++i, ++next) {
        for (int c = 0; c < 3; ++c) {
          REQUIRE(chunk(i, c) == expected(next, c));
        }
      }
      ++rounds;
    });
  REQUIRE(next == 7001);
  REQUIRE(candidates == markers.candidates);
  REQUIRE(rounds == candidates / 1000);

  // fewer samples than a batch
  const auto few = fusionio::SampleMarkers(
    f, domain, {.samples = 10, .batch = 1000, .seed = 9, .threshold = 1.5});
  REQUIRE(few.rows.extent(0) == 10);
  REQUIRE(few.candidates == 1000);
  const auto few_host = ToHost(few.rows);
  for (std::size_t i = 0; i < 10; ++i) {
    for (int c = 0; c < 3; ++c) {
      REQUIRE(few_host(i, c) == expected(i, c)); // the first of the candidates
    }
  }
}

TEST_CASE("SampleMarkers: invalid arguments", "[sampling]")
{
  const auto f = LinearFunction();
  const SamplingOptions options{.samples = 10, .batch = 10};
  CHECK_THROWS_WITH(fusionio::SampleMarkers(f, {{0, 1}}, options),
                    ContainsSubstring("coordinates"));
  CHECK_THROWS_WITH(fusionio::SampleMarkers(f, {{0, 1}, {1, 0}}, options),
                    ContainsSubstring("lo > hi"));
  CHECK_THROWS_WITH(
    fusionio::SampleMarkers(f, {{0, 1}, {0, 1}}, {.samples = 10, .batch = 0}),
    ContainsSubstring("batch"));
}

// The markers are a function of the seed alone: run with OMP_NUM_THREADS=1
// and 4 (the sampling_threads tests of CMakeLists.txt) writing them to the
// file named by FUSIONIO_TEST_MARKERS_FILE, and the files are compared.
TEST_CASE("SampleMarkers: markers to a file", "[threads]")
{
  const char* path = std::getenv("FUSIONIO_TEST_MARKERS_FILE");
  if (path == nullptr) {
    SUCCEED("FUSIONIO_TEST_MARKERS_FILE is not set");
    return;
  }
  const auto f = LinearFunction();
  const auto markers = fusionio::SampleMarkers(
    f, {{0, 1, Draw::Sqrt}, {0, 1}},
    {.samples = 5000, .batch = 2000, .seed = 11, .threshold = 1.7});
  const auto rows = ToHost(markers.rows);
  std::FILE* file = std::fopen(path, "w");
  REQUIRE(file != nullptr);
  for (std::size_t i = 0; i < rows.extent(0); ++i) {
    std::fprintf(file, "%a %a %a\n", rows(i, 0), rows(i, 1), rows(i, 2));
  }
  std::fclose(file);
}
