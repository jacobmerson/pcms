#include "bout_test_utils.hpp"
#include <pcms/transfer/l2_inner_product.hpp>
#include <algorithm>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::Real;

namespace
{

constexpr Real TwoPi = 2 * std::numbers::pi_v<Real>;

// A synthetic block: 4 x 8 interior points with one guard cell, 4 points in
// z over 2 pi / 5, zShift = c(x) y and a constant metric.
struct Synthetic
{
  static constexpr int Nx = 6;
  static constexpr int Ny = 10;
  static constexpr int Nz = 4;
  static constexpr Real Dx = 0.5;
  static constexpr Real Dy = 0.25;
  // g^xx, g^yy, g^zz, g^xy, g^xz, g^yz
  static constexpr Real G[6] = {4.0, 9.0, 1.0, 0.0, 0.5, 0.3};

  explicit Synthetic(Real shear)
  {
    for (int i = 0; i < Nx; ++i) {
      const Real c = 0.15 + shear * i;
      shift_angle.push_back(c * (Ny - 2));
      for (int j = 0; j < Ny; ++j) {
        const Real theta = TwoPi * (j - 1) / (Ny - 2);
        const Real r = 0.2 + 0.05 * i;
        rxy.push_back(3 + r * std::cos(theta));
        zxy.push_back(r * std::sin(theta));
        zshift.push_back(c * (j - 1));
      }
    }
    const auto view = [](const std::vector<Real>& a) {
      return fusionio::HostArray2D(a.data(), Nx, Ny);
    };
    grid.nx = Nx;
    grid.ny = Ny;
    grid.nz = Nz;
    grid.mxg = 1;
    grid.myg = 1;
    grid.zperiod = 5;
    grid.dz = TwoPi / (5 * Nz);
    grid.rxy = view(rxy);
    grid.zxy = view(zxy);
    grid.zshift = view(zshift);
    grid.dx = view(dx);
    grid.dy = view(dy);
    grid.shift_angle = fusionio::HostArray1D(shift_angle.data(), Nx);
    for (int c = 0; c < 6; ++c) {
      metric[c].assign(Nx * Ny, G[c]);
      grid.g_contra[c] = view(metric[c]);
    }
  }

  std::vector<Real> rxy, zxy, zshift, shift_angle;
  std::vector<Real> dx = std::vector<Real>(Nx * Ny, Dx);
  std::vector<Real> dy = std::vector<Real>(Nx * Ny, Dy);
  std::array<std::vector<Real>, 6> metric;
  fusionio::BoutGrid grid;
};

} // namespace

TEST_CASE("BOUT++ Build: positions, volume and the y seam of a synthetic grid",
          "[bout]")
{
  const Synthetic synthetic(0.0);
  const auto& grid = synthetic.grid;
  const auto spaces = fusionio::Build(grid);
  // (4 x 9 x 5) nodes: the y = ny and z = nz planes are DOF holders
  CHECK(spaces.space->GetLayout()->GetNumDofHolder() == 4 * 9 * 5);

  // the position at the points: (Rxy, z dz + zShift, Zxy)
  const auto pos = fusionio::test::AtIndex(*spaces.geometry->GetPosition(),
                                           {2, 3, 1, 4, 0, 3});
  CHECK_THAT(pos(0, 0), WithinAbs(grid.rxy(2, 4), 1e-14));
  CHECK_THAT(pos(0, 1), WithinAbs(grid.dz + grid.zshift(2, 4), 1e-14));
  CHECK_THAT(pos(0, 2), WithinAbs(grid.zxy(2, 4), 1e-14));
  CHECK_THAT(pos(1, 1), WithinAbs(3 * grid.dz + grid.zshift(4, 1), 1e-14));

  SECTION("the volume of the constant metric is J dx dy dz per point")
  {
    // J = 1 / sqrt(det g^ij); x by the trapezoid rule over the 4 points
    const auto& g = Synthetic::G;
    const Real det = g[0] * (g[1] * g[2] - g[5] * g[5]) -
                     g[3] * (g[3] * g[2] - g[5] * g[4]) +
                     g[4] * (g[3] * g[5] - g[1] * g[4]);
    const Real expected = 3 * 8 * Synthetic::Nz * Synthetic::Dx *
                          Synthetic::Dy * grid.dz / std::sqrt(det);
    auto one = spaces.space->CreateFunction<Real>("one");
    std::vector<Real> ones(Synthetic::Nx * Synthetic::Ny * Synthetic::Nz, 1.0);
    fusionio::CopyIn(ones, fusionio::InteriorOffsets(*spaces.space, grid), one);
    CHECK_THAT(pcms::L2Integral(spaces.space).Apply(one),
               WithinRel(expected, 1e-12));
  }

  SECTION("points past the y seam have the position of the twist-shifted "
          "points before it")
  {
    for (const Real x : {1.0, 2.3, 4.0}) {
      for (const Real z : {0.0, 1.5, 3.7}) {
        fusionio::test::CheckSeamPosition(spaces, 8, x, z, TwoPi / 5, 1e-5);
      }
    }
  }
}

TEST_CASE("BOUT++ Build: a twist-shift that varies in x", "[bout]")
{
  const Synthetic synthetic(0.04);
  const auto spaces = fusionio::Build(synthetic.grid);
  for (const Real x : {1.0, 2.3, 4.0}) {
    for (const Real z : {0.0, 1.5, 3.7}) {
      fusionio::test::CheckSeamPosition(spaces, 8, x, z, TwoPi / 5, 1e-5);
    }
  }
  // the shear transform of the metric on the y = ny plane keeps the volume
  // factor of the y = 0 plane
  const Real eps = 1e-10;
  const auto coords = fusionio::test::CreateDeviceCoordinateView(
    {2.5, 8 - eps, 1.5, 2.5, eps, 1.5}, spaces.geometry->GetParameterSystem(),
    3);
  const auto bound = spaces.geometry->Bind(coords.coordinate_view);
  const auto factor = bound->VolumeFactor();
  Kokkos::View<Real*, pcms::DeviceMemorySpace> volume("volume", 2);
  Kokkos::parallel_for(
    Kokkos::RangePolicy<pcms::DeviceMemorySpace::execution_space>(0, 2),
    KOKKOS_LAMBDA(const int p) { volume(p) = factor(p); });
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), volume);
  CHECK_THAT(host(0), WithinRel(host(1), 1e-8));
}

TEST_CASE("BOUT++ CopyIn and CopyOut: interior points and the upper y guard "
          "plane",
          "[bout]")
{
  const Synthetic synthetic(0.0);
  const auto& grid = synthetic.grid;
  const auto spaces = fusionio::Build(grid);
  const auto offsets = fusionio::InteriorOffsets(*spaces.space, grid);
  std::vector<Real> bout(Synthetic::Nx * Synthetic::Ny * Synthetic::Nz);
  for (std::size_t k = 0; k < bout.size(); ++k) {
    bout[k] = std::sin(0.7 * k);
  }
  auto u = spaces.space->CreateFunction<Real>("u");
  fusionio::CopyIn(bout, offsets, u);
  // BOUT++'s (x, y, z) = (2, 3, 1) is index point (2, 2, 1)
  // and (3, 9, 2), the upper guard, is the y = 8 plane before the seam
  const auto at = fusionio::test::AtIndex(u, {2, 2, 1, 3, 8 - 1e-13, 2});
  CHECK(at(0, 0) == bout[1 + Synthetic::Nz * (3 + Synthetic::Ny * 2)]);
  CHECK_THAT(
    at(1, 0),
    WithinAbs(bout[2 + Synthetic::Nz * (9 + Synthetic::Ny * 3)], 1e-11));

  std::vector<Real> out(bout.size(), 0.0);
  fusionio::CopyOut(u, offsets, out);
  for (int i = 1; i < Synthetic::Nx - 1; ++i) {
    for (int j = 1; j < Synthetic::Ny; ++j) {
      for (int k = 0; k < Synthetic::Nz; ++k) {
        const int o = k + Synthetic::Nz * (j + Synthetic::Ny * i);
        CHECK(out[o] == bout[o]);
      }
    }
  }
  CHECK(std::all_of(out.begin(), out.begin() + Synthetic::Nz * Synthetic::Ny,
                    [](Real v) { return v == 0; }));
  CHECK_THROWS(fusionio::CopyIn(std::span(bout).first(10), offsets, u));
}

TEST_CASE("BOUT++ Build: rejected grids", "[bout]")
{
  Synthetic synthetic(0.0);
  auto& grid = synthetic.grid;
  SECTION("open field lines")
  {
    grid.ixseps1 = 3;
    CHECK_THROWS(fusionio::Build(grid));
  }
  SECTION("blocks of a parallel run")
  {
    grid.x_offset = 4;
    CHECK_THROWS(fusionio::Build(grid));
  }
  SECTION("z not over the toroidal period")
  {
    grid.zperiod = 3;
    CHECK_THROWS(fusionio::Build(grid));
  }
  SECTION("offsets need a y guard cell")
  {
    const auto spaces = fusionio::Build(grid);
    grid.myg = 0;
    CHECK_THROWS(fusionio::InteriorOffsets(*spaces.space, grid));
  }
}
