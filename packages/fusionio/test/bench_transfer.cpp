// Benchmarks of field transfer from an M3D-C1 source: Fusion-IO 1.x called
// point by point; fusionio's bind, locate (QueryPoints) and evaluate, wrapping
// Fusion-IO 1.x and natively; and a PCMS Interpolator onto a grid. Hidden
// behind the [!benchmark] tag, so ctest does not run them; run them with
//   OMP_NUM_THREADS=1 fusionio_tests "[!benchmark]" --benchmark-samples 10
// (Fusion-IO 1.x is not thread safe; the native rows use the threads). Every
// point keeps its own search hint across samples, so the times are for warm
// hints, as in a repeated transfer.

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fusionio/fusionio.hpp>
#include <Kokkos_Core.hpp>
#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/field/coordinate_map.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/transfer/interpolator.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include "fusion_io_source.hpp"
#include "test_utils.hpp"
#include <limits>
#include <memory>
#include <string>
#include <vector>

using fusionio::Real;
using pcms::DeviceMemorySpace;

namespace
{

const std::string M3DC1File =
  std::string(FUSIONIO_TEST_DATA_DIR) + "/m3dc1/C1.h5";

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
  const fusionio::detail::FusionIOSourcePtr source(opened);
  fio_option_list options;
  source->get_field_options(&options);
  fio_field* field = nullptr;
  REQUIRE(source->get_field(type, &field, &options) == FIO_SUCCESS);
  const fusionio::detail::FusionIOFieldPtr owned(field);
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

/// fusionio: binds of the points, and evaluations of `quantity` at them
/// through a QueryPoints, whose Locate rebinds in place.
void BenchmarkFio(
  const std::string& name, fusionio::Quantity quantity,
  const std::vector<Real>& points,
  fusionio::Evaluation evaluation = fusionio::Evaluation::FusionIO)
{
  const auto source =
    fusionio::Open(fusionio::Format::M3DC1, M3DC1File, evaluation);
  const std::string label =
    evaluation == fusionio::Evaluation::Native ? "native" : "fusionio";
  const auto function = source->CreateFunction(quantity);
  const auto geometry = source->GetGeometry();
  const auto coords = fusionio::test::CreateDeviceCoordinateView(
    points, geometry->GetCoordinateSystem(), 3);
  const pcms::OutOfBoundsPolicy fill{pcms::OutOfBoundsMode::FILL,
                                     std::numeric_limits<Real>::quiet_NaN()};
  BENCHMARK(name + ": " + label + " bind (ERROR)")
  {
    return geometry->Bind(coords.coordinate_view);
  };
  BENCHMARK(name + ": " + label + " bind (FILL)")
  {
    return geometry->Bind(coords.coordinate_view, fill);
  };
  pcms::QueryPoints query(geometry, fill);
  query.Locate(coords.coordinate_view);
  BENCHMARK(name + ": " + label + " locate (FILL, QueryPoints)")
  {
    query.Locate(coords.coordinate_view);
    return query.Bound()->NumPoints();
  };
  Kokkos::View<Real**, DeviceMemorySpace> out(
    "out", points.size() / 3, function.GetLayout().GetNumComponents());
  BENCHMARK(name + ": " + label + " evaluate")
  {
    query.Evaluate(function, fusionio::test::TagLike<Real>(function, out));
    Kokkos::fence();
    return out.data()[0];
  };
}

} // namespace

TEST_CASE("Transfer benchmarks: M3D-C1 poloidal flux and magnetic field",
          "[!benchmark]")
{
  const auto points = GridPoints();
  BenchmarkFusionIO("psi", FIO_POLOIDAL_FLUX, points);
  BenchmarkFio("psi", fusionio::Quantity::PoloidalFlux, points);
#ifdef FUSIONIO_ENABLE_NATIVE
  BenchmarkFio("psi", fusionio::Quantity::PoloidalFlux, points,
               fusionio::Evaluation::Native);
#endif
  BenchmarkFusionIO("B", FIO_MAGNETIC_FIELD, points);
  BenchmarkFio("B", fusionio::Quantity::MagneticField, points);
#ifdef FUSIONIO_ENABLE_NATIVE
  BenchmarkFio("B", fusionio::Quantity::MagneticField, points,
               fusionio::Evaluation::Native);
#endif
}

TEST_CASE("Transfer benchmarks: M3D-C1 poloidal flux onto a P1 grid",
          "[!benchmark]")
{
  // The rectangle as a uniform grid of Divisions x Divisions cells; its
  // vertices are the grid points.
  pcms::UniformGrid<2> grid;
  grid.bot_left = {R0, Z0};
  grid.edge_length = {R1 - R0, Z1 - Z0};
  grid.divisions = {Divisions, Divisions};
  const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
  const auto psi = source->CreateFunction(fusionio::Quantity::PoloidalFlux);
  // The grid is the poloidal plane phi = 0 of the source's (R, phi, Z): the
  // map of the bind places its points there.
  const auto target = pcms::CreateLagrangeSpace(
    pcms::uniform_grid::CreateGeometry(grid, pcms::csys::CreateCylindricalRZ()),
    1, 1);
  const auto on_plane = pcms::CreatePoloidalPlaneEmbedding(0.0);
  const pcms::OutOfBoundsPolicy fill{pcms::OutOfBoundsMode::FILL,
                                     std::numeric_limits<Real>::quiet_NaN()};
  const auto bind = [&]() {
    return source->GetGeometry()->Bind(
      target->GetLayout()->GetDOFHolderCoordinates(), fill, on_plane);
  };
  REQUIRE(target->GetLayout()->GetNumDofHolder() ==
          (Divisions + 1) * (Divisions + 1));
  BENCHMARK("psi: Interpolator construction (FILL)")
  {
    return pcms::CreateInterpolator<Real>(psi.GetSpacePtr(), target, bind());
  };
  const auto interpolator =
    pcms::CreateInterpolator<Real>(psi.GetSpacePtr(), target, bind());
  auto psi_t = target->CreateFunction<Real>("psi");
  BENCHMARK("psi: Interpolator apply")
  {
    interpolator->Apply(psi, psi_t);
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
  BenchmarkFio("1 point, B", fusionio::Quantity::MagneticField, point);
  const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
  const auto b = source->CreateFunction(fusionio::Quantity::MagneticField);
  const auto bound = fusionio::test::Bind(*source->GetGeometry(), point);
  BENCHMARK("1 point, B: fusionio create evaluator")
  {
    return b.GetSpace().CreatePointEvaluator<Real>(bound);
  };
}
