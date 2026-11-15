// Benchmarks of field transfer from an M3D-C1 source: Fusion-IO 1.x called
// point by point, fio's bind and evaluate, and a PCMS Interpolator onto a
// mesh. Hidden behind the [!benchmark] tag, so ctest does not run them; run
// them with
//   OMP_NUM_THREADS=1 fusion_io_tests "[!benchmark]" --benchmark-samples 10
// (Fusion-IO 1.x is not thread safe). Every point keeps its own search hint
// across samples, so the times are for warm hints, as in a repeated transfer.

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fusion_io/fusion_io.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/field/coordinate_map.hpp>
#include <pcms/field/coordinate_systems/cylindrical.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/transformed_source.hpp>
#include <pcms/transfer/interpolator.h>
#include "fusion_io_source.hpp"
#include "test_utils.hpp"
#include <limits>
#include <memory>
#include <string>
#include <vector>

using fio::Real;
using pcms::DeviceMemorySpace;

namespace
{

const std::string M3DC1File =
  std::string(FUSION_IO_TEST_DATA_DIR) + "/m3dc1/C1.h5";

// A rectangle of the (R, Z) plane inside the example M3D-C1 domain, with
// Divisions + 1 points per side (99856 points).
constexpr Real R0 = 1.5, R1 = 1.9, Z0 = -0.3, Z1 = 0.3;
constexpr int Divisions = 315;

/// The grid points at phi = 0, (R, phi, Z) row-major.
std::vector<Real> GridPoints()
{
  std::vector<Real> points;
  for (int j = 0; j <= Divisions; ++j) {
    for (int i = 0; i <= Divisions; ++i) {
      points.insert(points.end(), {R0 + (R1 - R0) * i / Divisions, 0.0,
                                   Z0 + (Z1 - Z0) * j / Divisions});
    }
  }
  return points;
}

/// Fusion-IO 1.x evaluation of `type` at every point, one call per point
/// with a search hint per point, as in Fusion-IO's korc_example.
void BenchmarkFusionIO(const std::string& name, int type,
                       const std::vector<Real>& points)
{
  fio_source* opened = nullptr;
  REQUIRE(fio_open_source(&opened, FIO_M3DC1_SOURCE, M3DC1File.c_str()) ==
          FIO_SUCCESS);
  const fio::detail::FusionIOSourcePtr source(opened);
  fio_option_list options;
  source->get_field_options(&options);
  fio_field* field = nullptr;
  REQUIRE(source->get_field(type, &field, &options) == FIO_SUCCESS);
  const fio::detail::FusionIOFieldPtr owned(field);
  const size_t n = points.size() / 3;
  const auto dimension = static_cast<size_t>(field->dimension());
  const auto hint_size = static_cast<size_t>(source->sizeof_search_hint());
  std::vector<char> hints(hint_size * n, 0);
  std::vector<Real> values(dimension * n);
  BENCHMARK(name + ": Fusion-IO 1.x eval loop")
  {
    for (size_t i = 0; i < n; ++i) {
      field->eval(&points[3 * i], &values[dimension * i],
                  &hints[hint_size * i]);
    }
    return values[0];
  };
}

/// fio: one bind of the points, then evaluations of `quantity`.
void BenchmarkFio(const std::string& name, fio::Quantity quantity,
                  const std::vector<Real>& points)
{
  const auto source = fio::Open(fio::Format::M3DC1, M3DC1File);
  const auto function = source->CreateFunction(quantity);
  const auto& space = function.GetSpace();
  const auto coords = fio::test::CreateDeviceCoordinateView(
    points, space.GetCoordinateSystem(), 3);
  const pcms::OutOfBoundsPolicy fill{pcms::OutOfBoundsMode::FILL,
                                     std::numeric_limits<Real>::quiet_NaN()};
  BENCHMARK(name + ": fio bind (ERROR)")
  {
    return space.Bind(coords.coordinate_view);
  };
  BENCHMARK(name + ": fio bind (FILL)")
  {
    return space.Bind(coords.coordinate_view, fill);
  };
  const auto evaluator =
    space.CreatePointEvaluator<Real>(space.Bind(coords.coordinate_view, fill));
  Kokkos::View<Real**, DeviceMemorySpace> out(
    "out", points.size() / 3, function.GetLayout().GetNumComponents());
  BENCHMARK(name + ": fio evaluate")
  {
    evaluator->Evaluate(function, fio::test::TagLike<Real>(function, out));
    Kokkos::fence();
    return out.data()[0];
  };
}

/// The rectangle as an Omega_h triangle mesh with Divisions x Divisions
/// squares; its vertices are the grid points.
Omega_h::Mesh RectangleMesh(Omega_h::Library& lib)
{
  auto mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, R1 - R0, Z1 - Z0,
                                 0, Divisions, Divisions, 0, false);
  const Omega_h::HostRead<Omega_h::Real> unit(mesh.coords());
  Omega_h::HostWrite<Omega_h::Real> moved(unit.size());
  for (int i = 0; i < unit.size(); ++i) {
    moved[i] = unit[i] + (i % 2 == 0 ? R0 : Z0);
  }
  mesh.set_coords(Omega_h::Reals(moved.write()));
  return mesh;
}

} // namespace

TEST_CASE("Transfer benchmarks: M3D-C1 poloidal flux and magnetic field",
          "[!benchmark]")
{
  const auto points = GridPoints();
  BenchmarkFusionIO("psi", FIO_POLOIDAL_FLUX, points);
  BenchmarkFio("psi", fio::Quantity::PoloidalFlux, points);
  BenchmarkFusionIO("B", FIO_MAGNETIC_FIELD, points);
  BenchmarkFio("B", fio::Quantity::MagneticField, points);
}

TEST_CASE("Transfer benchmarks: M3D-C1 poloidal flux onto a P1 mesh",
          "[!benchmark]")
{
  auto lib = Omega_h::Library{};
  auto mesh = RectangleMesh(lib);
  const auto source = fio::Open(fio::Format::M3DC1, M3DC1File);
  const auto psi = source->CreateFunction(fio::Quantity::PoloidalFlux);
  // The mesh is the poloidal plane phi = 0 of the source's (R, phi, Z).
  auto target = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, pcms::csys::CylindricalRZ::Create(), "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
  const pcms::TransformedSource on_plane(
    psi.GetSpacePtr(), std::make_shared<pcms::PoloidalPlaneEmbedding>(0.0));
  const pcms::OutOfBoundsPolicy fill{pcms::OutOfBoundsMode::FILL,
                                     std::numeric_limits<Real>::quiet_NaN()};
  REQUIRE(target->GetLayout()->GetNumDofHolder() ==
          (Divisions + 1) * (Divisions + 1));
  BENCHMARK("psi: Interpolator construction (FILL)")
  {
    return pcms::Interpolator<Real>(on_plane, *target, fill);
  };
  const pcms::Interpolator<Real> interpolator(on_plane, *target, fill);
  auto psi_t = target->CreateFunction<Real>("psi");
  BENCHMARK("psi: Interpolator apply")
  {
    interpolator.Apply(psi, psi_t);
    Kokkos::fence();
    return &psi_t;
  };
}

TEST_CASE("Transfer benchmarks: fixed cost of one M3D-C1 point", "[!benchmark]")
{
  // One point per call, as in a tracer following a single field line: the
  // per-call cost of each step.
  const std::vector<Real> point = {1.75, 0.0, 0.05};
  BenchmarkFusionIO("1 point, B", FIO_MAGNETIC_FIELD, point);
  BenchmarkFio("1 point, B", fio::Quantity::MagneticField, point);
  const auto source = fio::Open(fio::Format::M3DC1, M3DC1File);
  const auto b = source->CreateFunction(fio::Quantity::MagneticField);
  const auto& space = b.GetSpace();
  const auto coords = fio::test::CreateDeviceCoordinateView(
    point, space.GetCoordinateSystem(), 3);
  const auto bound = space.Bind(coords.coordinate_view);
  BENCHMARK("1 point, B: fio create evaluator")
  {
    return space.CreatePointEvaluator<Real>(bound);
  };
}
