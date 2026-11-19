#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <pcms/field/coordinate_map.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/gradient_source.hpp>
#include <fusionio/fusionio.hpp>
#include "fusion_io_space.hpp"
#include "fusion_io_source.hpp"
#include <pcms/transfer/interpolator.hpp>
#include <pcms/eqdsk/eqdsk.hpp>
#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::test::EvaluateAt;
using fusionio::test::EvaluateGradientAt;
using pcms::DeviceMemorySpace;
using pcms::HostMemorySpace;
using pcms::Real;
using pcms::ValueBasis;

namespace
{

// The Fusion-IO example data (fusion-io/examples/data) in the test data.
const std::string GeqdskFile =
  std::string(FUSIONIO_TEST_DATA_DIR) + "/geqdsk/g158115.04701";
const std::string M3DC1File =
  std::string(FUSIONIO_TEST_DATA_DIR) + "/m3dc1/C1.h5";

// Fusion-IO's poloidal flux field is the total flux 2 pi R A_phi; geqdsk
// files and Fusion-IO's flux series are per radian.
constexpr Real TwoPi = 2.0 * std::numbers::pi;

Real EvaluateSeries(fio_source& source, int type)
{
  fio_series* opened = nullptr;
  const int status = source.get_series(type, &opened);
  const std::unique_ptr<fio_series> series(opened);
  REQUIRE(status == FIO_SUCCESS);
  Real value = 0.0;
  REQUIRE(series->eval(0.0, &value) == FIO_SUCCESS);
  return value;
}

} // namespace

TEST_CASE("Fusion-IO geqdsk poloidal flux matches the grid values")
{
  const auto psi = fusionio::Open(fusionio::Format::GEQDSK, GeqdskFile)
                     ->CreateFunction(fusionio::Quantity::PoloidalFlux);

  // Fusion-IO's interpolant reproduces the grid values (read here by PCMS's
  // own reader) at the grid nodes, and phi must not matter.
  const pcms::EQDSKData eqdsk(GeqdskFile);
  std::vector<Real> points;
  std::vector<Real> expected;
  for (int i : {eqdsk.GetNW() / 3, eqdsk.GetNW() / 2}) {
    for (int j : {eqdsk.GetNH() / 3, eqdsk.GetNH() / 2}) {
      for (Real phi : {0.0, 1.3}) {
        points.insert(points.end(), {eqdsk.GetR(i), phi, eqdsk.GetZ(j)});
        expected.push_back(TwoPi * eqdsk.GetPsi(i, j));
      }
    }
  }
  const auto result = EvaluateAt(psi, points);
  for (size_t i = 0; i < expected.size(); ++i) {
    CAPTURE(i, points[3 * i], points[3 * i + 2]);
    REQUIRE_THAT(result(i, 0), WithinRel(expected[i], fusionio::test::EvalTol));
  }
}

TEST_CASE("Fusion-IO M3D-C1 fields")
{
  auto source =
    fusionio::detail::OpenFusionIOSource(fusionio::Format::M3DC1, M3DC1File);
  const auto opened = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
  auto psi = opened->CreateFunction(fusionio::Quantity::PoloidalFlux);
  auto a = opened->CreateFunction(fusionio::Quantity::VectorPotential);
  const ValueBasis basis = a.GetData().GetValueBasis();
  REQUIRE(*basis.system == *pcms::csys::CreateCylindricalRThetaZ());
  REQUIRE(basis.scaling == pcms::ComponentScaling::Physical);
  REQUIRE(basis.variance == pcms::values::Vector);

  // The binding stores each point's search hint as sizeof_search_hint()
  // zero-initialized bytes (an element guess for M3D-C1). Revisit it if this
  // changes.
  REQUIRE(source->Get().sizeof_search_hint() == sizeof(int));

  SECTION("the flux at the magnetic axis is the axis flux")
  {
    const Real r = EvaluateSeries(source->Get(), FIO_MAGAXIS_R);
    const Real z = EvaluateSeries(source->Get(), FIO_MAGAXIS_Z);
    const auto result = EvaluateAt(psi, {r, 0.0, z});
    REQUIRE_THAT(
      result(0, 0),
      WithinRel(TwoPi * EvaluateSeries(source->Get(), FIO_MAGAXIS_PSI), 1e-3));
  }

  SECTION("vector components are in (R, phi, Z) order: psi = 2 pi R A_phi")
  {
    const std::vector<Real> points = {1.6, 0.1, 0.2, 2.0, 2.5, -0.4};
    const auto psi_values = EvaluateAt(psi, points);
    const auto a_values = EvaluateAt(a, points);
    for (size_t i = 0; i < psi_values.extent(0); ++i) {
      CAPTURE(i);
      REQUIRE_THAT(TwoPi * points[3 * i] * a_values(i, 1),
                   WithinRel(psi_values(i, 0), fusionio::test::EvalTol));
    }
  }

  SECTION("(R, phi, Z) is right-handed: B_R = -dA_phi/dZ")
  {
    // The axisymmetric part has no phi derivative, so this holds exactly for
    // a right-handed system and with the opposite sign otherwise.
    const fusionio::Options equilibrium{.part = fusionio::Part::Equilibrium};
    auto a_eq =
      opened->CreateFunction(fusionio::Quantity::VectorPotential, equilibrium);
    auto b_eq =
      opened->CreateFunction(fusionio::Quantity::MagneticField, equilibrium);
    const Real r = 2.0, z = 0.3, dz = 1e-4;
    const auto a_values = EvaluateAt(a_eq, {r, 0.0, z + dz, r, 0.0, z - dz});
    const auto b_values = EvaluateAt(b_eq, {r, 0.0, z});
    const Real dadz = (a_values(0, 1) - a_values(1, 1)) / (2.0 * dz);
    // The central difference is accurate to O(dz^2), far below dz.
    REQUIRE_THAT(b_values(0, 0), WithinRel(-dadz, dz));
  }

  SECTION("options reach Fusion-IO: the equilibrium part is axisymmetric")
  {
    auto psi_eq = opened->CreateFunction(fusionio::Quantity::PoloidalFlux,
                                         {.part = fusionio::Part::Equilibrium});
    const std::vector<Real> points = {1.6, 0.1, 0.2, 1.6, 2.5, 0.2};
    const auto total = EvaluateAt(psi, points);
    const auto equilibrium = EvaluateAt(psi_eq, points);
    REQUIRE_THAT(equilibrium(1, 0),
                 WithinRel(equilibrium(0, 0), fusionio::test::ExactTol));
    // The perturbation in the total varies with phi.
    REQUIRE_FALSE(total(1, 0) == total(0, 0));
  }

  SECTION("evaluators created from one bound source share its search")
  {
    const std::vector<Real> points = {1.6, 0.1, 0.2, 2.0, 2.5, -0.4};
    const auto bound = fusionio::test::Bind(*opened->GetGeometry(), points);
    auto psi_eq = opened->CreateFunction(fusionio::Quantity::PoloidalFlux,
                                         {.part = fusionio::Part::Equilibrium});
    auto b = opened->CreateFunction(fusionio::Quantity::MagneticField);
    for (const auto* f : {&psi, &psi_eq, &b}) {
      const auto shared = fusionio::test::EvaluateWith(
        *f->GetSpace().CreatePointEvaluator<Real>(bound), *f);
      const auto separate = EvaluateAt(*f, points);
      for (size_t i = 0; i < 2; ++i) {
        for (size_t c = 0; c < shared.extent(1); ++c) {
          CAPTURE(f->GetName(), i, c);
          REQUIRE(shared(i, c) == separate(i, c));
        }
      }
    }
  }

  SECTION("a bound source rebinds in place and keeps evaluating correctly")
  {
    const std::vector<Real> first = {1.6, 0.1, 0.2, 2.0, 2.5, -0.4};
    const std::vector<Real> second = {1.3,  0.0, 0.4, 1.6, 0.7,
                                      -0.6, 2.0, 2.5, 0.1};
    const auto& geometry = *opened->GetGeometry();
    auto bound = fusionio::test::Bind(geometry, first);
    const auto* storage = bound.get();
    const auto coords = fusionio::test::CreateDeviceCoordinateView(
      second, geometry.GetCoordinateSystem(), 3);
    bound = geometry.Rebind(std::move(bound), coords.coordinate_view);
    REQUIRE(bound.get() == storage);
    const auto rebound = fusionio::test::EvaluateWith(
      *psi.GetSpace().CreatePointEvaluator<Real>(bound), psi);
    const auto fresh = EvaluateAt(psi, second);
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      REQUIRE(rebound(i, 0) == fresh(i, 0));
    }
  }

  SECTION("values match direct fusion-io evaluation")
  {
    const fusionio::Options options{.timeslice = 1,
                                    .part = fusionio::Part::Perturbed};
    fio_option_list fio_options;
    source->Get().get_field_options(&fio_options);
    fio_options.set_option(FIO_TIMESLICE, 1);
    fio_options.set_option(FIO_PART, FIO_PERTURBED_ONLY);
    const std::vector<Real> points = {1.3, 0.0, 0.4, 1.6, 0.7, -0.6,
                                      2.0, 2.5, 0.1, 2.2, 4.0, -0.3};
    for (const auto quantity : {fusionio::Quantity::PoloidalFlux,
                                fusionio::Quantity::MagneticField}) {
      CAPTURE(fusionio::Name(quantity));
      const auto values =
        EvaluateAt(opened->CreateFunction(quantity, options), points);
      const auto field = source->GetField(quantity, fio_options);
      for (size_t i = 0; i < values.extent(0); ++i) {
        Real expected[3];
        REQUIRE(field->eval(&points[3 * i], expected) == FIO_SUCCESS);
        for (int c = 0; c < field->dimension(); ++c) {
          CAPTURE(i, c);
          // The absolute floor covers components near zero.
          REQUIRE_THAT(values(i, c),
                       WithinRel(expected[c], fusionio::test::ExactTol) ||
                         WithinAbs(expected[c], fusionio::test::ExactTol));
        }
      }
    }
  }
}

namespace
{

// A source reporting Cartesian geometry; nothing else is used.
class CartesianSource final : public fio_source
{
public:
  int open(const char*) override { return FIO_SUCCESS; }
  int close() override { return FIO_SUCCESS; }
  int get_field_options(fio_option_list*) const override { return FIO_SUCCESS; }
  int get_field(const field_type, fio_field**, const fio_option_list*) override
  {
    return FIO_UNSUPPORTED;
  }
  int get_available_fields(fio_field_list*) const override
  {
    return FIO_SUCCESS;
  }
  int get_int_parameter(const parameter_type t, int* p) const override
  {
    if (t != FIO_GEOMETRY) {
      return FIO_UNSUPPORTED;
    }
    *p = FIO_CARTESIAN;
    return FIO_SUCCESS;
  }
};

} // namespace

TEST_CASE("Fusion-IO evaluation errors")
{
  const auto source = fusionio::Open(fusionio::Format::GEQDSK, GeqdskFile);
  auto psi = source->CreateFunction(fusionio::Quantity::PoloidalFlux);

  // geqdsk sources locate points only under FILL; evaluation reports the
  // points outside under ERROR.
  REQUIRE_THROWS_WITH(EvaluateAt(psi, {50.0, 0.0, 0.0}),
                      ContainsSubstring("outside the source's domain"));
  REQUIRE_THROWS_WITH(source->CreateFunction(fusionio::Quantity::Viscosity),
                      ContainsSubstring("\"viscosity\" is not a quantity") &&
                        ContainsSubstring("poloidal flux"));
  REQUIRE_THROWS_WITH(psi.GetDOFHolderData(),
                      ContainsSubstring("no DOFAccess"));
  REQUIRE_THROWS_WITH(psi.GetSpace().CreateFunction<Real>("psi"),
                      ContainsSubstring("does not name a quantity"));
  REQUIRE_THROWS_WITH(
    fusionio::detail::OpenFusionIOSource(fusionio::Format::GEQDSK, "missing"),
    ContainsSubstring("could not open"));
  REQUIRE_THROWS_WITH(
    fusionio::detail::FusionIOSource(
      fusionio::Format::M3DC1,
      fusionio::detail::FusionIOSourcePtr(new CartesianSource)),
    ContainsSubstring("cylindrical geometry"));

  // Fusion-IO writes one row per query point.
  const auto evaluator =
    psi.GetSpace().CreatePointEvaluator<Real>(fusionio::test::Bind(
      *source->GetGeometry(), {1.8, 0.0, 0.1, 2.0, 0.0, 0.1}));
  Kokkos::View<Real**, DeviceMemorySpace> short_out("short_out", 1, 1);
  REQUIRE_THROWS_WITH(
    evaluator->Evaluate(psi, fusionio::test::TagLike(psi, short_out)),
    ContainsSubstring("but must be [2][1]"));

  // Spaces of different sources do not share bound sources or fields.
  const auto other = fusionio::Open(fusionio::Format::GEQDSK, GeqdskFile);
  auto other_psi = other->CreateFunction(fusionio::Quantity::PoloidalFlux);
  REQUIRE_THROWS_WITH(
    other_psi.GetSpace().CreatePointEvaluator<Real>(
      fusionio::test::Bind(*source->GetGeometry(), {1.8, 0.0, 0.1})),
    ContainsSubstring("another geometry"));
  const auto other_evaluator = other_psi.GetSpace().CreatePointEvaluator<Real>(
    fusionio::test::Bind(*other->GetGeometry(), {1.8, 0.0, 0.1}));
  Kokkos::View<Real**, DeviceMemorySpace> out("out", 1, 1);
  REQUIRE_THROWS_WITH(
    other_evaluator->Evaluate(psi, fusionio::test::TagLike(psi, out)),
    ContainsSubstring("not a field of the function space"));
}

TEST_CASE("Fusion-IO FILL finds the points the domain probe evaluates")
{
  // M3D-C1 sources locate points with a mesh search instead of evaluating
  // the probe, and geqdsk sources evaluate it: both must find the points
  // where the probe evaluates, here on a grid across the domain boundary.
  const auto [format, file, r0, r1, z0, z1] =
    GENERATE(table<fusionio::Format, std::string, Real, Real, Real, Real>(
      {{fusionio::Format::M3DC1, M3DC1File, 0.8, 2.8, -1.6, 1.6},
       {fusionio::Format::GEQDSK, GeqdskFile, 0.5, 3.0, -2.0, 2.0}}));
  CAPTURE(file);
  auto source = fusionio::detail::OpenFusionIOSource(format, file);
  const auto opened = fusionio::Open(format, file);
  auto b = opened->CreateFunction(fusionio::Quantity::MagneticField);
  constexpr int N = 24;
  std::vector<Real> points;
  for (int j = 0; j <= N; ++j) {
    for (int i = 0; i <= N; ++i) {
      points.insert(points.end(),
                    {r0 + (r1 - r0) * i / N, 0.4 * j, z0 + (z1 - z0) * j / N});
    }
  }
  const size_t n = points.size() / 3;
  std::vector<pcms::LO> outside;
  std::vector<Real> probe_values(3);
  for (size_t i = 0; i < n; ++i) {
    if (source->DomainProbe().eval(&points[3 * i], probe_values.data()) ==
        FIO_OUT_OF_BOUNDS) {
      outside.push_back(static_cast<pcms::LO>(i));
    }
  }
  REQUIRE(!outside.empty());
  REQUIRE(outside.size() < n);

  constexpr Real Fill = -7.0;
  const auto evaluator =
    b.GetSpace().CreatePointEvaluator<Real>(fusionio::test::Bind(
      *opened->GetGeometry(), points, {pcms::OutOfBoundsMode::FILL, Fill}));
  const auto filled = Kokkos::create_mirror_view_and_copy(
    HostMemorySpace(), evaluator->FilledPoints());
  REQUIRE(std::vector<pcms::LO>(filled.data(),
                                filled.data() + filled.extent(0)) == outside);
  Kokkos::View<Real**, DeviceMemorySpace> out("out", n, 3);
  evaluator->Evaluate(b, fusionio::test::TagLike(b, out));
  const auto values =
    Kokkos::create_mirror_view_and_copy(HostMemorySpace(), out);
  for (const pcms::LO i : outside) {
    REQUIRE(values(i, 0) == Fill);
  }
}

TEST_CASE("Fusion-IO gradients")
{
  SECTION("M3D-C1: the equilibrium field is the curl of the flux")
  {
    const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
    const fusionio::Options equilibrium{.part = fusionio::Part::Equilibrium};
    auto psi =
      source->CreateFunction(fusionio::Quantity::PoloidalFlux, equilibrium);
    auto b =
      source->CreateFunction(fusionio::Quantity::MagneticField, equilibrium);
    // Axisymmetric, so B_R = -dpsi/dZ / (2 pi R) and B_Z = dpsi/dR / (2 pi R)
    // for the total flux psi, and the toroidal derivative vanishes.
    const std::vector<Real> points = {1.6,  0.1, 0.2, 2.0, 2.5,
                                      -0.4, 1.3, 4.0, 0.5};
    const auto grad = EvaluateGradientAt(psi, points);
    const auto b_values = EvaluateAt(b, points);
    for (size_t i = 0; i < grad.extent(0); ++i) {
      const Real r = points[3 * i];
      CAPTURE(i);
      REQUIRE_THAT(-grad(i, 2) / (TwoPi * r),
                   WithinRel(b_values(i, 0), fusionio::test::EvalTol));
      REQUIRE_THAT(grad(i, 0) / (TwoPi * r),
                   WithinRel(b_values(i, 2), fusionio::test::EvalTol));
      REQUIRE_THAT(grad(i, 1), WithinAbs(0.0, fusionio::test::ExactTol));
    }
    // Fusion-IO provides the derivatives of B, but the package does not
    // evaluate the gradients of vectors yet.
    REQUIRE(b.GetSpace().Supports(pcms::DifferentialOperator::Gradient));
    REQUIRE_THROWS_WITH(
      b.GetSpace().CreatePointEvaluator<Real>(
        fusionio::test::Bind(*source->GetGeometry(), points),
        pcms::DifferentialOperator::Gradient),
      ContainsSubstring("more than one component are not supported"));
  }

  SECTION("geqdsk: the gradient matches finite differences of the flux")
  {
    auto psi = fusionio::Open(fusionio::Format::GEQDSK, GeqdskFile)
                 ->CreateFunction(fusionio::Quantity::PoloidalFlux);
    // Inside a grid cell: Fusion-IO's bicubic interpolant of the geqdsk grid
    // is not differentiable across grid lines, where the finite difference
    // would straddle two cells.
    const Real r = 1.9, phi = 0.7, z = 0.263, h = 1e-5;
    const auto grad = EvaluateGradientAt(psi, {r, phi, z});
    const auto values = EvaluateAt(
      psi, {r + h, phi, z, r - h, phi, z, r, phi, z + h, r, phi, z - h});
    // The central differences are accurate to O(h^2) relative to the gradient.
    REQUIRE_THAT(grad(0, 0),
                 WithinRel((values(0, 0) - values(1, 0)) / (2.0 * h), 1e-6));
    REQUIRE_THAT(grad(0, 2),
                 WithinRel((values(2, 0) - values(3, 0)) / (2.0 * h), 1e-6));
    REQUIRE_THAT(grad(0, 1), WithinAbs(0.0, fusionio::test::ExactTol));
  }

  SECTION("M3D-C1: the flux gradient interpolates onto a grid")
  {
    const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
    const fusionio::Options equilibrium{.part = fusionio::Part::Equilibrium};
    auto psi =
      source->CreateFunction(fusionio::Quantity::PoloidalFlux, equilibrium);
    auto b =
      source->CreateFunction(fusionio::Quantity::MagneticField, equilibrium);

    // A grid in (R, theta, Z) inside the M3D-C1 domain.
    pcms::UniformGrid<3> grid;
    grid.bot_left = {1.5, 0.0, -0.3};
    grid.edge_length = {0.4, 1.0, 0.6};
    grid.divisions = {3, 2, 3};
    const auto target = pcms::uniform_grid::CreateGeometry(
      grid, pcms::csys::CreateCylindricalRThetaZ());
    auto value_space = pcms::CreateLagrangeSpace(target, 1, 1);
    auto vector_space = pcms::CreateLagrangeSpace(target, 1, 3);
    auto psi_t = value_space->CreateFunction<Real>("psi");
    const auto& system = target->GetCoordinateSystem();
    auto grad_psi_t = vector_space->CreateFunction<Real>(
      "grad psi", pcms::ValueBasis{system, pcms::ComponentScaling::Physical,
                                   pcms::values::Covector});
    auto b_t = vector_space->CreateFunction<Real>(
      "B", pcms::ValueBasis{system, pcms::ComponentScaling::Physical,
                            pcms::values::Vector});

    // The flux, its gradient and the field share one search of the grid
    // points.
    const auto bound = source->GetGeometry()->Bind(
      value_space->GetLayout()->GetDOFHolderCoordinates());
    pcms::CreateInterpolator<Real>(psi.GetSpacePtr(), value_space, bound)
      ->Apply(psi, psi_t);
    pcms::CreateInterpolator<Real>(pcms::Gradient(psi.GetSpacePtr()),
                                   vector_space, bound)
      ->Apply(psi, grad_psi_t);
    pcms::CreateInterpolator<Real>(b.GetSpacePtr(), vector_space, bound)
      ->Apply(b, b_t);

    // Axisymmetric: B_R = -dpsi/dZ / (2 pi R), B_Z = dpsi/dR / (2 pi R).
    const auto coords = fusionio::test::CopyCoordinatesToHost(
      value_space->GetLayout()->GetDOFHolderCoordinates().GetValues());
    const auto grad = grad_psi_t.GetDOFHolderDataHost().GetValues();
    const auto field = b_t.GetDOFHolderDataHost().GetValues();
    REQUIRE(coords.extent(0) == 4 * 3 * 4);
    for (size_t i = 0; i < coords.extent(0); ++i) {
      const Real r = coords(i, 0);
      CAPTURE(i, r, coords(i, 1), coords(i, 2));
      REQUIRE_THAT(-grad(i, 2) / (TwoPi * r),
                   WithinRel(field(i, 0), fusionio::test::EvalTol));
      REQUIRE_THAT(grad(i, 0) / (TwoPi * r),
                   WithinRel(field(i, 2), fusionio::test::EvalTol));
      REQUIRE_THAT(grad(i, 1), WithinAbs(0.0, fusionio::test::ExactTol));
    }
  }

  SECTION("gradients are re-expressed in Cartesian components")
  {
    const auto source = fusionio::Open(fusionio::Format::M3DC1, M3DC1File);
    auto psi = source->CreateFunction(fusionio::Quantity::PoloidalFlux);
    const Real r = 1.7, phi = 2.2, z = 0.3;
    const auto cyl = EvaluateGradientAt(psi, {r, phi, z});
    const Real expected[3] = {
      cyl(0, 0) * std::cos(phi) - cyl(0, 1) * std::sin(phi),
      cyl(0, 0) * std::sin(phi) + cyl(0, 1) * std::cos(phi), cyl(0, 2)};
    // The points are mapped to the source's (R, phi, Z) by the bind, and the
    // gradient is the space's operator or an operator source over it.
    const auto coords = fusionio::test::CreateDeviceCoordinateView(
      {r * std::cos(phi), r * std::sin(phi), z}, pcms::csys::CreateCartesian(3),
      3);
    const auto bound = source->GetGeometry()->Bind(
      coords.coordinate_view, {}, pcms::CreateCartesianToCylindrical());
    std::vector<std::unique_ptr<pcms::PointEvaluator<Real>>> evaluators;
    evaluators.push_back(psi.GetSpace().CreatePointEvaluator<Real>(
      bound, pcms::DifferentialOperator::Gradient));
    evaluators.push_back(
      pcms::Gradient(psi.GetSpacePtr())->CreatePointEvaluator<Real>(bound));
    for (const auto& evaluator : evaluators) {
      const ValueBasis basis =
        evaluator->OutputBasis(psi.GetData().GetValueBasis());
      // Physical components on an orthonormal basis: PCMS's cylindrical
      // rotation reports them as a vector's, which are the same numbers.
      REQUIRE(basis.Rank() == 1);
      REQUIRE(basis.scaling == pcms::ComponentScaling::Physical);
      REQUIRE(*basis.system == *pcms::csys::CreateCartesian(3));
      const auto host = fusionio::test::EvaluateWith(*evaluator, psi);
      for (int c = 0; c < 3; ++c) {
        CAPTURE(c);
        REQUIRE_THAT(host(0, c),
                     WithinRel(expected[c], fusionio::test::EvalTol) ||
                       WithinAbs(expected[c], fusionio::test::ExactTol));
      }
    }
  }
}
