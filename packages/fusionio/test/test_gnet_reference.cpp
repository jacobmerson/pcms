// Compares with the output of the original gnet2gtc (PCMS branch
// gnet-gtc-coupling) when FUSIONIO_GNET_REFERENCE_DIR names a directory with
// the original's input synthetic_gnet.h5 (make_synthetic_gnet's file) and its
// output gtc_ref.h5 (/gtc_n6d):
//   OMP_NUM_THREADS=1 gnet2gtc --input synthetic_gnet.h5 --output gtc_ref.h5
//     --samples 20000 --batch 5000 --seed 14000 --threshold-scale 1
// The original's candidates come from a Kokkos random pool that depends on
// the thread count, so they are not reproduced: this function is evaluated at
// the candidates of the reference (columns 0-4) and compared with its values
// (column 5).
#include <catch2/catch_test_macros.hpp>
#include "fusionio/gnet.hpp"
#include <hdf5.h>
#include <pcms/field/query_points.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

using pcms::Real;

namespace
{

constexpr int Dim = 5;

// the rows of /gtc_n6d
std::vector<std::array<Real, Dim + 1>> ReadTable(const std::string& path)
{
  const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  REQUIRE(file >= 0);
  const hid_t dataset = H5Dopen(file, "/gtc_n6d", H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  const hid_t space = H5Dget_space(dataset);
  hsize_t dims[2] = {};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  REQUIRE(dims[1] == Dim + 1);
  std::vector<std::array<Real, Dim + 1>> rows(dims[0]);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          rows.data());
  H5Sclose(space);
  H5Dclose(dataset);
  H5Fclose(file);
  return rows;
}

std::vector<Real> Evaluate(const pcms::Function<Real>& f,
                           const std::vector<std::array<Real, Dim + 1>>& rows)
{
  const std::size_t n = rows.size();
  Kokkos::View<Real**, pcms::HostMemorySpace> host("points", n, Dim);
  for (std::size_t i = 0; i < n; ++i) {
    for (int d = 0; d < Dim; ++d) {
      host(i, d) = rows[i][d];
    }
  }
  Kokkos::View<Real**, pcms::DeviceMemorySpace> device("points", n, Dim);
  pcms::DeepCopyMismatchLayouts(device, host);
  Kokkos::View<Real**, pcms::DeviceMemorySpace> out("values", n, 1);
  pcms::Evaluate(
    f,
    pcms::CoordinateView<pcms::DeviceMemorySpace>(
      f.GetSpace().GetCoordinateSystem(), pcms::MakeConstRank2View(device)),
    pcms::ValueView<Real, pcms::DeviceMemorySpace>(f.GetData().GetValueBasis(),
                                                   pcms::MakeRank2View(out)));
  const auto host_out =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
  std::vector<Real> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = host_out(i, 0);
  }
  return values;
}

Real RelativeDifference(Real a, Real b)
{
  return std::abs(a - b) / std::max(std::abs(b), 1e-300);
}

// Whether the supports of the point wrap a periodic bin: the bin search takes
// the bin below the point and the one above it (wrapping at the last), so the
// wrapped bin is used for the points above the centre of the last bin, in the
// poloidal bins of the radial bins around the point, and in the toroidal
// ones.
bool NearSeam(const std::array<Real, Dim + 1>& row, int num_rho, int num_zeta,
              Real zeta_max)
{
  const int rho = std::max(0, static_cast<int>(num_rho * row[0] - 0.5));
  bool near = false;
  // the radial bins around the point: the one below and the one above, or the
  // one below the last (the search moves to the other side at an end)
  for (const int r : {rho, rho + 1 < num_rho ? rho + 1 : rho - 1}) {
    const Real width = 2 * std::numbers::pi / (4 * (r + 1));
    near = near || row[1] > 2 * std::numbers::pi - width / 2;
  }
  return near || row[2] > zeta_max - zeta_max / num_zeta / 2;
}

// The acceptance of the full rounds of a run of this package's sampling.
double Acceptance(const pcms::Function<Real>& f, Real threshold)
{
  std::vector<std::size_t> takes;
  fusionio::SampleMarkers(
    f, fusionio::GTCDomain(),
    {.samples = 100000, .batch = 20000, .seed = 14000, .threshold = threshold},
    [&](const fusionio::MarkerRows& rows) { takes.push_back(rows.extent(0)); });
  takes.pop_back(); // the last round is clipped
  std::size_t accepted = 0;
  for (const auto take : takes) {
    accepted += take;
  }
  return static_cast<double>(accepted) / (takes.size() * 20000);
}

} // namespace

TEST_CASE("GNET: comparison with the output of the original gnet2gtc",
          "[gnet][reference]")
{
  const char* dir = std::getenv("FUSIONIO_GNET_REFERENCE_DIR");
  if (dir == nullptr) {
    SUCCEED("FUSIONIO_GNET_REFERENCE_DIR is not set");
    return;
  }
  const std::string input = std::string(dir) + "/synthetic_gnet.h5";
  const auto rows = ReadTable(std::string(dir) + "/gtc_ref.h5");
  REQUIRE(rows.size() == 20000);

  // step A: the raw coordinates of the original
  const auto raw = fusionio::OpenGnet(input, {.periodic = false});
  const auto raw_values = Evaluate(raw.density, rows);
  Real max_relative = 0;
  Real max_absolute = 0;
  std::vector<Real> relative;
  std::size_t below_threshold = 0;
  const Real threshold = 1.0 / raw.max_volume;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    relative.push_back(RelativeDifference(raw_values[i], rows[i][5]));
    max_relative = std::max(max_relative, relative.back());
    max_absolute = std::max(max_absolute, std::abs(raw_values[i] - rows[i][5]));
    below_threshold += raw_values[i] < threshold;
  }
  std::sort(relative.begin(), relative.end());
  const Real median = relative[relative.size() / 2];
  std::printf("step A: relative difference median %.3e, max %.3e (absolute "
              "%.3e); %zu of %zu rows below the threshold\n",
              median, max_relative, max_absolute, below_threshold, rows.size());
  // the same supports and fit, so equal to round-off, amplified by the
  // conditioning of the fit (energy is of order 1e5, pitch of order 1)
  CHECK(median < 1e-10);
  CHECK(max_relative < 1e-6);

  // the acceptance: the reference's 8 rounds of 5000 accepted 18353 of the
  // 35000 candidates of its first 7 (0.5244 +- 0.0027)
  const double acceptance = Acceptance(raw.density, threshold);
  std::printf("acceptance: reference %.4f (+- 0.0027), here %.4f\n",
              18353.0 / 35000, acceptance);
  CHECK(std::abs(acceptance - 18353.0 / 35000) < 0.015);
  CHECK(below_threshold == 0);

  // step B: periodic theta and zeta
  const auto periodic = fusionio::OpenGnet(input, {.periodic = true});
  const auto periodic_values = Evaluate(periodic.density, rows);
  Real max_away = 0;
  Real max_seam = 0;
  std::size_t seam = 0;
  std::size_t changed = 0;
  std::size_t changed_away = 0;
  Real sum_changed = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const Real difference =
      RelativeDifference(periodic_values[i], raw_values[i]);
    if (NearSeam(rows[i], 4, 3, raw.max_zeta)) {
      ++seam;
      max_seam = std::max(max_seam, difference);
    } else {
      max_away = std::max(max_away, difference);
    }
    if (difference > 1e-9) {
      ++changed;
      sum_changed += difference;
      changed_away += !NearSeam(rows[i], 4, 3, raw.max_zeta);
    }
  }
  std::printf("step B: %zu of %zu rows near a seam; max relative change near "
              "%.3e, away %.3e; %zu changed (> 1e-9, mean change %.3e), %zu "
              "of them away; acceptance %.4f\n",
              seam, rows.size(), max_seam, max_away, changed,
              sum_changed / std::max<std::size_t>(changed, 1), changed_away,
              Acceptance(periodic.density, threshold));
  CHECK(max_away < 1e-9);
  CHECK(changed_away == 0);
}
