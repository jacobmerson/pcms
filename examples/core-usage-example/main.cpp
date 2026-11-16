// Uses PCMS core through its installed public headers only, without Omega_h
// or MPI: builds the geometry of a uniform grid and a linear Lagrange space on
// it, and evaluates a linear function at points with QueryPoints.

#include <pcms/field/function_spaces.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/geometry/geometry.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <pcms/utility/print.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace
{
constexpr pcms::Real LinearTol = 1e-10;

pcms::Real Linear(pcms::Real x, pcms::Real y)
{
  return 1.0 + 2.0 * x + 3.0 * y;
}
} // namespace

int main(int argc, char** argv)
{
  int mismatches = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    const auto cartesian = pcms::csys::Cartesian::Create(2);
    const pcms::UniformGrid<2> grid{
      .edge_length = {1.0, 1.0}, .bot_left = {0.0, 0.0}, .divisions = {4, 4}};
    auto geometry = pcms::CreateUniformGridGeometry(grid, cartesian);
    auto space = pcms::CreateLagrangeSpace(geometry, 1, 1);
    auto source = space->CreateFunction<pcms::Real>("source");

    // source(x, y) = 1 + 2x + 3y at the DOF holders: linear interpolation
    // reproduces it exactly.
    const auto dof_coords = source.GetLayout().GetDOFHolderCoordinates();
    const auto num_dofs = dof_coords.GetValues().extent(0);
    Kokkos::View<pcms::Real**, pcms::HostMemorySpace> coords_host("coords_host",
                                                                  num_dofs, 2);
    auto coords_mirror =
      Kokkos::create_mirror_view(pcms::DeviceMemorySpace(), coords_host);
    pcms::ConvertMismatchLayoutView2D(coords_mirror, dof_coords.GetValues());
    Kokkos::deep_copy(coords_host, coords_mirror);
    std::vector<pcms::Real> values(num_dofs);
    for (std::size_t i = 0; i < num_dofs; ++i) {
      values[i] = Linear(coords_host(i, 0), coords_host(i, 1));
    }
    source.SetDOFHolderDataHost(
      pcms::ValueView<const pcms::Real, pcms::HostMemorySpace>(
        source.GetData().GetValueBasis(),
        pcms::Rank2View<const pcms::Real, pcms::HostMemorySpace>(values.data(),
                                                                 num_dofs, 1)));

    const std::vector<pcms::Real> points{0.3, 0.4, 0.75, 0.1, 0.9, 0.9};
    const std::size_t num_points = points.size() / 2;
    Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> point_data(
      "points", num_points, 2);
    auto point_host = Kokkos::create_mirror_view(point_data);
    for (std::size_t i = 0; i < num_points; ++i) {
      point_host(i, 0) = points[2 * i];
      point_host(i, 1) = points[2 * i + 1];
    }
    Kokkos::deep_copy(point_data, point_host);
    pcms::QueryPoints query(geometry);
    query.Locate(pcms::CoordinateView<pcms::DeviceMemorySpace>(
      cartesian, pcms::MakeRank2View(point_data)));
    Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> out("out", num_points,
                                                            1);
    query.Evaluate(
      source, pcms::ValueView<pcms::Real, pcms::DeviceMemorySpace>(
                source.GetData().GetValueBasis(), pcms::MakeRank2View(out)));
    const auto out_host =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
    for (std::size_t i = 0; i < num_points; ++i) {
      if (std::abs(out_host(i, 0) - Linear(points[2 * i], points[2 * i + 1])) >
          LinearTol) {
        ++mismatches;
      }
    }
    pcms::printInfo("evaluated %d values, %d mismatches\n",
                    static_cast<int>(num_points), mismatches);
  }
  return mismatches == 0 ? 0 : 1;
}
