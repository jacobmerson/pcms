#include "fusionio/bout.hpp"
#include "bout_points.hpp"
#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/periodic_axis.hpp>
#include <pcms/field/value_view.hpp>
#include <pcms/utility/assert.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include <pcms/utility/coordinate_systems/parameter.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace fusionio
{

namespace
{

using detail::BlockPoint;
using detail::BlockPoints;
using pcms::HostMemorySpace;
using pcms::LO;

constexpr Real TwoPi = 2 * std::numbers::pi_v<Real>;

int InteriorNx(const BoutGrid& grid)
{
  return grid.nx - 2 * grid.mxg;
}

int InteriorNy(const BoutGrid& grid)
{
  return grid.ny - 2 * grid.myg;
}

void Check(const BoutGrid& grid)
{
  const bool closed =
    grid.ixseps1 >= grid.nx && grid.ixseps2 >= grid.nx && grid.jyseps1_1 < 0 &&
    grid.jyseps2_2 >= InteriorNy(grid) - 1 && grid.jyseps1_2 == grid.jyseps2_1;
  if (!closed) {
    throw pcms::pcms_error("fusionio::Build: only grids of closed field lines "
                           "are supported (open field lines: see the "
                           "separatrix indices)");
  }
  if (grid.x_offset != 0 || grid.y_offset != 0) {
    throw pcms::pcms_error(
      "fusionio::Build: only one serial block is supported (offsets 0)");
  }
  if (grid.mxg < 0 || grid.myg < 0 || InteriorNx(grid) < 2 ||
      InteriorNy(grid) < 2 || grid.nz < 2 || grid.zperiod < 1) {
    throw pcms::pcms_error("fusionio::Build: needs at least 2 interior points "
                           "in x, y and z and zperiod >= 1");
  }
  const auto fits = [&](const HostArray2D& a) {
    return a.extent(0) == grid.nx && a.extent(1) == grid.ny;
  };
  const bool arrays = fits(grid.rxy) && fits(grid.zxy) && fits(grid.zshift) &&
                      fits(grid.dx) && fits(grid.dy) &&
                      std::ranges::all_of(grid.g_contra, fits) &&
                      grid.shift_angle.extent(0) == grid.nx;
  if (!arrays) {
    throw pcms::pcms_error(
      "fusionio::Build: the arrays must be [nx][ny] (shift_angle [nx])");
  }
  if (std::abs(grid.nz * grid.dz - TwoPi / grid.zperiod) >
      1e-10 * TwoPi / grid.zperiod) {
    throw pcms::pcms_error("fusionio::Build: nz dz must be 2 pi / zperiod");
  }
}

} // namespace

std::vector<BlockPoint> detail::BlockPoints(const pcms::FunctionSpace& space,
                                            const BoutGrid& grid)
{
  const auto coords = space.GetLayout()->GetDOFHolderCoordinates().GetValues();
  Kokkos::View<Real**, pcms::DeviceMemorySpace> device(
    "bout_dof_holders", coords.extent(0), coords.extent(1));
  pcms::ConvertMismatchLayoutView2D(device, coords);
  const auto host =
    Kokkos::create_mirror_view_and_copy(HostMemorySpace(), device);
  std::vector<BlockPoint> points(coords.extent(0));
  for (std::size_t k = 0; k < points.size(); ++k) {
    const auto x = static_cast<int>(std::lround(host(k, 0))) - grid.x_offset;
    const auto y = static_cast<int>(std::lround(host(k, 1))) - grid.y_offset;
    const auto z = static_cast<int>(std::lround(host(k, 2)));
    if (x < grid.mxg || x >= grid.nx - grid.mxg || y < 0 ||
        y > InteriorNy(grid) || z < 0 || z > grid.nz) {
      throw pcms::pcms_error(
        "fusionio: a DOF holder is not a point of the BOUT++ block");
    }
    const bool seam = y == InteriorNy(grid);
    points[k] = {x, grid.myg + (seam ? 0 : y), z, seam};
  }
  return points;
}

namespace
{

/// d(ShiftAngle / dz) / dx in index units at local x `i`.
Real ShiftSlope(const BoutGrid& grid, int i)
{
  const int lo = std::max(i - 1, grid.mxg);
  const int hi = std::min(i + 1, grid.nx - grid.mxg - 1);
  return (grid.shift_angle(hi) - grid.shift_angle(lo)) / ((hi - lo) * grid.dz);
}

/// g^xx, g^yy, g^zz, g^xy, g^xz, g^yz in index units at a point of the block.
void IndexMetric(const BoutGrid& grid, const BlockPoint& p, Real* g)
{
  const Real h[3] = {grid.dx(p.ix, p.jy), grid.dy(p.ix, p.jy), grid.dz};
  constexpr int First[6] = {0, 1, 2, 0, 0, 1};
  constexpr int Second[6] = {0, 1, 2, 1, 2, 2};
  for (int c = 0; c < 6; ++c) {
    g[c] = grid.g_contra[c](p.ix, p.jy) / (h[First[c]] * h[Second[c]]);
  }
}

template <typename Fill>
std::shared_ptr<const pcms::Function<Real>> NodeFunction(
  const std::shared_ptr<const pcms::FunctionSpace>& space, const BoutGrid& grid,
  const std::string& name, pcms::ValueBasis basis, Fill fill)
{
  auto function = std::make_shared<pcms::Function<Real>>(
    space->CreateFunction<Real>(name, std::move(basis)));
  const auto points = BlockPoints(*space, grid);
  Kokkos::View<Real**, HostMemorySpace> values(
    name, points.size(), space->GetLayout()->GetNumComponents());
  for (std::size_t k = 0; k < points.size(); ++k) {
    fill(points[k], &values(k, 0));
  }
  function->SetDOFHolderDataHost(pcms::ValueView<const Real, HostMemorySpace>(
    function->GetData().GetValueBasis(), pcms::MakeConstRank2View(values)));
  return function;
}

} // namespace

BoutSpaces Build(const BoutGrid& grid, BoutMetric metric)
{
  Check(grid);
  const int nxi = InteriorNx(grid);
  const int nyi = InteriorNy(grid);
  const Real x0 = grid.x_offset + grid.mxg;

  // the twist-shift S(x) = ShiftAngle / dz, in z index units
  pcms::UniformGrid<1> radial;
  radial.bot_left = {x0};
  radial.edge_length = {static_cast<Real>(nxi - 1)};
  radial.divisions = {nxi - 1};
  const auto radial_geometry = pcms::uniform_grid::CreateGeometry(
    radial, pcms::csys::CreateParameterSystem("bout_x", 1));
  const auto radial_space = pcms::CreateLagrangeSpace(radial_geometry, 1, 1);
  const auto radial_function = [&](const std::string& name, Real scale) {
    auto function = std::make_shared<pcms::Function<Real>>(
      radial_space->CreateFunction<Real>(name));
    Kokkos::View<Real**, HostMemorySpace> values(name, nxi, 1);
    for (int i = 0; i < nxi; ++i) {
      values(i, 0) = scale * grid.shift_angle(grid.mxg + i) / grid.dz;
    }
    function->SetDOFHolderDataHost(pcms::ValueView<const Real, HostMemorySpace>(
      function->GetData().GetValueBasis(), pcms::MakeConstRank2View(values)));
    return std::shared_ptr<const pcms::Function<Real>>(std::move(function));
  };
  // z keeps its last node: a plain periodic axis over the grid would
  // drop it, and phi would then be interpolated from z = nz - 1 back to 0
  const auto index = pcms::csys::CreateParameterSystem(
    "bout_index", 3,
    {{1, 0.0, static_cast<Real>(nyi),
      pcms::CreateTwistShiftTransform(2, radial_function("shift_angle", 1),
                                      {0})},
     {2, 0.0, static_cast<Real>(grid.nz),
      pcms::CreateTwistShiftTransform(2, radial_function("zero", 0), {0})}});

  pcms::UniformGrid<3> cells;
  cells.bot_left = {x0, 0.0, 0.0};
  cells.edge_length = {static_cast<Real>(nxi - 1), static_cast<Real>(nyi),
                       static_cast<Real>(grid.nz)};
  cells.divisions = {nxi - 1, nyi, grid.nz};
  const auto geometry = pcms::uniform_grid::CreateGeometry(cells, index);

  const auto position =
    NodeFunction(pcms::CreateLagrangeSpace(geometry, 1, 3), grid, "position",
                 pcms::ValueBasis::Coordinates(
                   pcms::csys::CreateCylindricalRThetaZ(TwoPi / grid.zperiod)),
                 [&](const BlockPoint& p, Real* out) {
                   const Real shift = grid.zshift(p.ix, p.jy) +
                                      (p.seam ? grid.shift_angle(p.ix) : 0);
                   out[0] = grid.rxy(p.ix, p.jy);
                   out[1] = p.kz * grid.dz + shift;
                   out[2] = grid.zxy(p.ix, p.jy);
                 });
  // g^ij in index components; on the y = ny plane in the z of the plane
  // before the seam, z - S(x)
  pcms::EmbeddingOptions options;
  if (metric == BoutMetric::Given) {
    options.metric =
      NodeFunction(pcms::CreateLagrangeSpace(geometry, 1, 9), grid, "g^ij",
                   {.scaling = pcms::ComponentScaling::Natural,
                    .variance = pcms::values::Tensor},
                   [&](const BlockPoint& p, Real* out) {
                     Real g[6];
                     IndexMetric(grid, p, g);
                     if (p.seam) {
                       const Real s = ShiftSlope(grid, p.ix);
                       g[2] += s * (s * g[0] - 2 * g[4]);
                       g[4] -= s * g[0];
                       g[5] -= s * g[3];
                     }
                     const Real full[9] = {g[0], g[3], g[4], g[3], g[1],
                                           g[5], g[4], g[5], g[2]};
                     std::copy(std::begin(full), std::end(full), out);
                   });
  }
  // J = 1 / sqrt(det g^ij); the seam's shear has determinant 1
  const auto volume = [&](const BlockPoint& p, Real* out) {
    Real g[6];
    IndexMetric(grid, p, g);
    const Real det = g[0] * (g[1] * g[2] - g[5] * g[5]) -
                     g[3] * (g[3] * g[2] - g[5] * g[4]) +
                     g[4] * (g[3] * g[5] - g[1] * g[4]);
    out[0] = 1 / std::sqrt(det);
  };
  options.volume = NodeFunction(pcms::CreateLagrangeSpace(geometry, 1, 1), grid,
                                "J", {}, volume);
  auto embedded =
    pcms::CreateEmbeddedGeometry(geometry, position, std::move(options));
  auto space = pcms::CreateEmbeddedSpace(
    embedded, pcms::CreateLagrangeSpace(geometry, 1, 1));
  return {std::move(embedded), std::move(space)};
}

std::vector<LO> InteriorOffsets(const pcms::FunctionSpace& space,
                                const BoutGrid& grid)
{
  if (grid.myg < 1) {
    throw pcms::pcms_error("fusionio::InteriorOffsets: the y = ny plane "
                           "reads the first upper y guard plane: myg >= 1");
  }
  const auto points = BlockPoints(space, grid);
  std::vector<LO> offsets(points.size());
  for (std::size_t k = 0; k < points.size(); ++k) {
    const auto& p = points[k];
    const int jy = p.seam ? grid.myg + InteriorNy(grid) : p.jy;
    offsets[k] = p.kz % grid.nz + grid.nz * (jy + grid.ny * p.ix);
  }
  return offsets;
}

void CopyIn(std::span<const Real> values, std::span<const LO> offsets,
            pcms::Function<Real>& function)
{
  const auto n = static_cast<LO>(offsets.size());
  if (function.GetData().GetValueBasis().Rank() != 0 ||
      std::ranges::any_of(offsets, [&](LO o) {
        return o < 0 || static_cast<std::size_t>(o) >= values.size();
      })) {
    throw pcms::pcms_error("fusionio::CopyIn: needs a scalar function and "
                           "offsets within the values");
  }
  Kokkos::View<Real**, HostMemorySpace> dofs("bout_copy_in", n, 1);
  for (LO k = 0; k < n; ++k) {
    dofs(k, 0) = values[offsets[k]];
  }
  function.SetDOFHolderDataHost(pcms::ValueView<const Real, HostMemorySpace>(
    function.GetData().GetValueBasis(), pcms::MakeConstRank2View(dofs)));
}

void CopyOut(const pcms::Function<Real>& function, std::span<const LO> offsets,
             std::span<Real> values)
{
  const auto dofs = function.GetDOFHolderDataHost().GetValues();
  if (dofs.extent(1) != 1 ||
      dofs.extent(0) != static_cast<LO>(offsets.size()) ||
      std::ranges::any_of(offsets, [&](LO o) {
        return o < 0 || static_cast<std::size_t>(o) >= values.size();
      })) {
    throw pcms::pcms_error("fusionio::CopyOut: needs a scalar function and "
                           "an offset within the values per DOF holder");
  }
  for (std::size_t k = 0; k < offsets.size(); ++k) {
    values[offsets[k]] = dofs(static_cast<LO>(k), 0);
  }
}

} // namespace fusionio
