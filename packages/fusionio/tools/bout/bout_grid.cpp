// bout-grid: a BOUT++ grid file as PCMS objects.
//
// Reads the grid, builds its geometry and prints its volume (by quadrature
// with the grid's metric and as BOUT++'s sum of J dx dy dz), the integral of
// the pressure and the position and pressure at an index point.

#include <fusionio/bout.hpp>
#include <pcms/transfer/l2_inner_product.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <exception>
#include <iostream>
#include <string>

namespace
{

using fusionio::Real;

// the values of `function` at the index point `point`
std::vector<Real> At(const pcms::Function<Real>& function,
                     const std::vector<Real>& point)
{
  const auto& geometry = *function.GetSpace().GetGeometry();
  const auto system = geometry.GetParameterSystem()
                        ? geometry.GetParameterSystem()
                        : geometry.GetCoordinateSystem();
  Kokkos::View<Real**, pcms::HostMemorySpace> host("point", 1, 3);
  for (int c = 0; c < 3; ++c) {
    host(0, c) = point[c];
  }
  Kokkos::View<Real**, pcms::DeviceMemorySpace> device("point", 1, 3);
  pcms::DeepCopyMismatchLayouts(device, host);
  const auto evaluator = function.GetSpace().CreatePointEvaluator<Real>(
    geometry.Bind(pcms::CoordinateView<pcms::DeviceMemorySpace>(
      system, pcms::MakeRank2View(device))));
  const auto basis = evaluator->OutputBasis(function.GetData().GetValueBasis());
  const int columns = function.GetLayout().GetNumComponents();
  Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", 1, columns);
  evaluator->Evaluate(function, pcms::ValueView<Real, pcms::DeviceMemorySpace>(
                                  basis, pcms::MakeRank2View(out)));
  Kokkos::View<Real**, pcms::HostMemorySpace> values("values", 1, columns);
  pcms::DeepCopyMismatchLayouts(values, out);
  return {values.data(), values.data() + columns};
}

int Run(const std::string& path, int nz, int zperiod)
{
  const auto file = fusionio::ReadBoutGrid(path, nz, zperiod);
  const auto& grid = file.grid;
  const auto& a = file.arrays;
  Real sum = 0;
  for (int x = grid.mxg; x < grid.nx - grid.mxg; ++x) {
    const Real weight =
      (x == grid.mxg || x == grid.nx - grid.mxg - 1) ? 0.5 : 1.0;
    for (int y = 0; y < grid.ny; ++y) {
      const auto p = static_cast<std::size_t>(x) * grid.ny + y;
      sum += weight * a.at("hthe")[p] / a.at("Bpxy")[p] * a.at("dx")[p] *
             a.at("dy")[p];
    }
  }
  sum *= nz * grid.dz;

  const auto source = fusionio::OpenBout(path, nz, zperiod);
  const auto pressure =
    source->CreateFunction(fusionio::Quantity::TotalPressure);
  auto one = pressure.GetSpace().CreateFunction<Real>("one");
  Kokkos::View<Real**, pcms::HostMemorySpace> ones(
    "one", one.GetLayout().GetNumDofHolder(), 1);
  Kokkos::deep_copy(ones, 1.0);
  one.SetDOFHolderDataHost(pcms::ValueView<const Real, pcms::HostMemorySpace>(
    one.GetData().GetValueBasis(), pcms::MakeConstRank2View(ones)));
  const pcms::L2Integral integral(pressure.GetSpace().shared_from_this());

  const auto& embedded =
    dynamic_cast<const pcms::EmbeddedGeometry&>(*source->GetGeometry());
  const std::vector<Real> point = {grid.nx / 2.0, grid.ny / 4.0, 0.0};
  const auto position = At(*embedded.GetPosition(), point);
  std::cout << "grid " << grid.nx << " x " << grid.ny << " x " << nz
            << ", x guard cells " << grid.mxg << ", toroidal period "
            << source->ToroidalPeriod() << " rad\n"
            << "volume (m^3): quadrature " << integral.Apply(one)
            << ", sum of J dx dy dz " << sum << "\n"
            << "integral of the pressure (J): " << integral.Apply(pressure)
            << "\n"
            << "at (x, y, z) = (" << point[0] << ", " << point[1] << ", "
            << point[2] << "): (R, phi, Z) = (" << position[0] << ", "
            << position[1] << ", " << position[2]
            << "), pressure (Pa) = " << At(pressure, point)[0] << "\n";
  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <grid.nc> <nz> <zperiod>\n";
    return 1;
  }
  MPI_Init(&argc, &argv);
  int result = 1;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      result = Run(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));
    } catch (const std::exception& e) {
      std::cerr << "bout-grid: " << e.what() << "\n";
    }
  }
  MPI_Finalize();
  return result;
}
