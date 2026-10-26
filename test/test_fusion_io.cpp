#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <pcms/field/coordinate_map.hpp>
#include <pcms/field/coordinate_systems/cartesian.hpp>
#include <pcms/field/coordinate_systems/cylindrical.hpp>
#include <pcms/field/gradient_source.hpp>
#include <pcms/field/transformed_source.hpp>
#include <pcms/field/function_space/fusion_io.hpp>
#include <pcms/field/function_space/fusion_io_source.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/transfer/interpolator.h>
#include <pcms/utility/eqdsk.h>
#include "field_test_utils.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using pcms::DeviceMemorySpace;
using pcms::FusionIOFormat;
using pcms::FusionIOOptions;
using pcms::HostMemorySpace;
using pcms::Real;
using pcms::ValueBasis;

namespace
{

// The Fusion-IO example data (fusion-io/examples/data) in the test data.
const std::string GeqdskFile =
  std::string(PCMS_TEST_FUSIONIO_DIR) + "/geqdsk/g158115.04701";
const std::string M3DC1File =
  std::string(PCMS_TEST_FUSIONIO_DIR) + "/m3dc1/C1.h5";

// Fusion-IO's poloidal flux field is the total flux 2 pi R A_phi; geqdsk
// files and Fusion-IO's flux series are per radian.
constexpr Real TwoPi = 2.0 * std::numbers::pi;

// Values of `field` at `points`, (R, phi, Z) row-major, copied to the host.
auto EvaluateAt(const pcms::FunctionSpace& space,
                const pcms::Field<Real>& field, const std::vector<Real>& points)
{
  const auto coords = pcms::test::CreateDeviceCoordinateView(
    points, space.GetCoordinateSystem(), 3);
  const auto evaluator =
    space.CreatePointEvaluator<Real>(space.Bind(coords.coordinate_view));
  Kokkos::View<Real**, DeviceMemorySpace> out(
    "out", points.size() / 3, field.GetLayout().GetNumComponents());
  evaluator->Evaluate(field, pcms::test::TagLike(field, out));
  return Kokkos::create_mirror_view_and_copy(HostMemorySpace(), out);
}

// Gradient of the scalar `field` at `points`, (dR, (1/R) dphi, dZ) per row.
auto EvaluateGradientAt(const pcms::FunctionSpace& space,
                        const pcms::Field<Real>& field,
                        const std::vector<Real>& points)
{
  const auto coords = pcms::test::CreateDeviceCoordinateView(
    points, space.GetCoordinateSystem(), 3);
  const auto evaluator =
    space.CreatePointEvaluator<Real>(space.Bind(coords.coordinate_view), 1);
  Kokkos::View<Real**, DeviceMemorySpace> out("out", points.size() / 3, 3);
  evaluator->Evaluate(field,
                      pcms::ValueView<Real, DeviceMemorySpace>(
                        evaluator->OutputBasis(field.GetData().GetValueBasis()),
                        pcms::MakeRank2View(out)));
  return Kokkos::create_mirror_view_and_copy(HostMemorySpace(), out);
}

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
  auto source = pcms::OpenFusionIOSource(FusionIOFormat::GEQDSK, GeqdskFile);
  auto space = pcms::FusionIOFunctionSpace::Create(source, 1);
  auto psi = space->CreateFunction("psi", "poloidal flux");

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
  const auto result = EvaluateAt(*space, psi, points);
  for (size_t i = 0; i < expected.size(); ++i) {
    CAPTURE(i, points[3 * i], points[3 * i + 2]);
    REQUIRE_THAT(result(i, 0), WithinRel(expected[i], pcms::test::EvalTol));
  }
}

TEST_CASE("Fusion-IO M3D-C1 fields")
{
  auto source = pcms::OpenFusionIOSource(FusionIOFormat::M3DC1, M3DC1File);
  auto scalars = pcms::FusionIOFunctionSpace::Create(source, 1);
  auto vectors = pcms::FusionIOFunctionSpace::Create(source, 3);
  auto psi = scalars->CreateFunction("psi", "poloidal flux");
  auto a = vectors->CreateFunction("A", "vector potential");
  REQUIRE(pcms::SameValueBasis(
    a.GetData().GetValueBasis(),
    ValueBasis{pcms::csys::CylindricalRThetaZ::Create(),
               pcms::ComponentScaling::Physical, pcms::values::Vector}));

  // The evaluator stores each point's search hint as sizeof_search_hint()
  // zero-initialized bytes (an element guess for M3D-C1). Revisit it if this
  // changes.
  REQUIRE(source->Get().sizeof_search_hint() == sizeof(int));

  SECTION("the flux at the magnetic axis is the axis flux")
  {
    const Real r = EvaluateSeries(source->Get(), FIO_MAGAXIS_R);
    const Real z = EvaluateSeries(source->Get(), FIO_MAGAXIS_Z);
    const auto result = EvaluateAt(*scalars, psi, {r, 0.0, z});
    REQUIRE_THAT(
      result(0, 0),
      WithinRel(TwoPi * EvaluateSeries(source->Get(), FIO_MAGAXIS_PSI), 1e-3));
  }

  SECTION("vector components are in (R, phi, Z) order: psi = 2 pi R A_phi")
  {
    const std::vector<Real> points = {1.6, 0.1, 0.2, 2.0, 2.5, -0.4};
    const auto psi_values = EvaluateAt(*scalars, psi, points);
    const auto a_values = EvaluateAt(*vectors, a, points);
    for (size_t i = 0; i < psi_values.extent(0); ++i) {
      CAPTURE(i);
      REQUIRE_THAT(TwoPi * points[3 * i] * a_values(i, 1),
                   WithinRel(psi_values(i, 0), pcms::test::EvalTol));
    }
  }

  SECTION("(R, phi, Z) is right-handed: B_R = -dA_phi/dZ")
  {
    // The axisymmetric part has no phi derivative, so this holds exactly for
    // a right-handed system and with the opposite sign otherwise.
    const FusionIOOptions equilibrium{.part =
                                        FusionIOOptions::Part::Equilibrium};
    auto a_eq = vectors->CreateFunction("A", "vector potential", equilibrium);
    auto b_eq = vectors->CreateFunction("B", "magnetic field", equilibrium);
    const Real r = 2.0, z = 0.3, dz = 1e-4;
    const auto a_values =
      EvaluateAt(*vectors, a_eq, {r, 0.0, z + dz, r, 0.0, z - dz});
    const auto b_values = EvaluateAt(*vectors, b_eq, {r, 0.0, z});
    const Real dadz = (a_values(0, 1) - a_values(1, 1)) / (2.0 * dz);
    // The central difference is accurate to O(dz^2), far below dz.
    REQUIRE_THAT(b_values(0, 0), WithinRel(-dadz, dz));
  }

  SECTION("options reach Fusion-IO: the equilibrium part is axisymmetric")
  {
    auto psi_eq = scalars->CreateFunction(
      "psi_eq", "poloidal flux", {.part = FusionIOOptions::Part::Equilibrium});
    const std::vector<Real> points = {1.6, 0.1, 0.2, 1.6, 2.5, 0.2};
    const auto total = EvaluateAt(*scalars, psi, points);
    const auto equilibrium = EvaluateAt(*scalars, psi_eq, points);
    REQUIRE_THAT(equilibrium(1, 0),
                 WithinRel(equilibrium(0, 0), pcms::test::ExactTol));
    // The perturbation in the total varies with phi.
    REQUIRE_FALSE(total(1, 0) == total(0, 0));
  }

  SECTION("evaluators created from one bound source share its search")
  {
    const std::vector<Real> points = {1.6, 0.1, 0.2, 2.0, 2.5, -0.4};
    const auto coords = pcms::test::CreateDeviceCoordinateView(
      points, scalars->GetCoordinateSystem(), 3);
    const auto bound = scalars->Bind(coords.coordinate_view);
    auto psi_eq = scalars->CreateFunction(
      "psi_eq", "poloidal flux", {.part = FusionIOOptions::Part::Equilibrium});
    for (const auto* f : {&psi, &psi_eq}) {
      const auto evaluator = scalars->CreatePointEvaluator<Real>(bound);
      Kokkos::View<Real**, DeviceMemorySpace> out("out", 2, 1);
      evaluator->Evaluate(*f, pcms::test::TagLike(*f, out));
      const auto shared =
        Kokkos::create_mirror_view_and_copy(HostMemorySpace(), out);
      const auto separate = EvaluateAt(*scalars, *f, points);
      for (size_t i = 0; i < 2; ++i) {
        CAPTURE(f->GetName(), i);
        REQUIRE_THAT(shared(i, 0),
                     WithinRel(separate(i, 0), pcms::test::ExactTol));
      }
    }
    REQUIRE_THROWS_WITH(vectors->CreatePointEvaluator<Real>(bound),
                        ContainsSubstring("bound by another factory"));
  }

  SECTION("values match direct fusion-io evaluation")
  {
    const FusionIOOptions options{.timeslice = 1,
                                  .part = FusionIOOptions::Part::Perturbed};
    fio_option_list fio_options;
    source->Get().get_field_options(&fio_options);
    fio_options.set_option(FIO_TIMESLICE, 1);
    fio_options.set_option(FIO_PART, FIO_PERTURBED_ONLY);
    const std::vector<Real> points = {1.3, 0.0, 0.4, 1.6, 0.7, -0.6,
                                      2.0, 2.5, 0.1, 2.2, 4.0, -0.3};
    for (const auto& [quantity, space] :
         {std::pair{"poloidal flux", scalars},
          std::pair{"magnetic field", vectors}}) {
      CAPTURE(quantity);
      const auto values = EvaluateAt(
        *space, space->CreateFunction("f", quantity, options), points);
      const auto field =
        source->GetField(source->Quantities().at(quantity), fio_options);
      for (size_t i = 0; i < values.extent(0); ++i) {
        Real expected[3];
        REQUIRE(field->eval(&points[3 * i], expected) == FIO_SUCCESS);
        for (int c = 0; c < field->dimension(); ++c) {
          CAPTURE(i, c);
          // The absolute floor covers components near zero.
          REQUIRE_THAT(values(i, c),
                       WithinRel(expected[c], pcms::test::ExactTol) ||
                         WithinAbs(expected[c], pcms::test::ExactTol));
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
  auto source = pcms::OpenFusionIOSource(FusionIOFormat::GEQDSK, GeqdskFile);
  auto scalars = pcms::FusionIOFunctionSpace::Create(source, 1);
  auto vectors = pcms::FusionIOFunctionSpace::Create(source, 3);
  auto psi = scalars->CreateFunction("psi", "poloidal flux");
  const auto quantities = pcms::AvailableQuantities(*source);
  REQUIRE(std::find(quantities.begin(), quantities.end(), "magnetic field") !=
          quantities.end());

  REQUIRE_THROWS_WITH(EvaluateAt(*scalars, psi, {50.0, 0.0, 0.0}),
                      ContainsSubstring("outside the source's domain"));
  REQUIRE_THROWS_WITH(scalars->CreateFunction("B", "magnetic field"),
                      ContainsSubstring("the space holds 1"));
  REQUIRE_THROWS_WITH(vectors->CreateFunction("psi", "poloidal flux"),
                      ContainsSubstring("the space holds 3"));
  REQUIRE_THROWS_WITH(
    scalars->CreateFunction("psi", "poloidal fluxx"),
    ContainsSubstring("\"poloidal fluxx\" is not a quantity") &&
      ContainsSubstring("poloidal flux,"));
  REQUIRE_THROWS_WITH(psi.GetDOFHolderData(),
                      ContainsSubstring("no DOFAccess"));
  REQUIRE_THROWS_WITH(
    pcms::OpenFusionIOSource(FusionIOFormat::GEQDSK, "missing"),
    ContainsSubstring("could not open"));
  REQUIRE_THROWS_WITH(
    pcms::FusionIOSource(pcms::FusionIOSourcePtr(new CartesianSource)),
    ContainsSubstring("cylindrical geometry"));

  // Fusion-IO writes one row per query point.
  const auto coords = pcms::test::CreateDeviceCoordinateView(
    {1.8, 0.0, 0.1, 2.0, 0.0, 0.1}, scalars->GetCoordinateSystem(), 3);
  const auto evaluator =
    scalars->CreatePointEvaluator<Real>(scalars->Bind(coords.coordinate_view));
  Kokkos::View<Real**, DeviceMemorySpace> short_out("short_out", 1, 1);
  REQUIRE_THROWS_WITH(
    evaluator->Evaluate(psi, pcms::test::TagLike(psi, short_out)),
    ContainsSubstring("but expected [2][1]"));

  auto other = pcms::FusionIOFunctionSpace::Create(
    pcms::OpenFusionIOSource(FusionIOFormat::GEQDSK, GeqdskFile), 1);
  REQUIRE_THROWS_WITH(EvaluateAt(*other, psi, {1.8, 0.0, 0.1}),
                      ContainsSubstring("field is not a quantity"));
}

TEST_CASE("Fusion-IO gradients")
{
  SECTION("M3D-C1: the equilibrium field is the curl of the flux")
  {
    auto source = pcms::OpenFusionIOSource(FusionIOFormat::M3DC1, M3DC1File);
    auto scalars = pcms::FusionIOFunctionSpace::Create(source, 1);
    auto vectors = pcms::FusionIOFunctionSpace::Create(source, 3);
    const FusionIOOptions equilibrium{.part =
                                        FusionIOOptions::Part::Equilibrium};
    auto psi = scalars->CreateFunction("psi", "poloidal flux", equilibrium);
    auto b = vectors->CreateFunction("B", "magnetic field", equilibrium);
    // Axisymmetric, so B_R = -dpsi/dZ / (2 pi R) and B_Z = dpsi/dR / (2 pi R)
    // for the total flux psi, and the toroidal derivative vanishes.
    const std::vector<Real> points = {1.6,  0.1, 0.2, 2.0, 2.5,
                                      -0.4, 1.3, 4.0, 0.5};
    const auto grad = EvaluateGradientAt(*scalars, psi, points);
    const auto b_values = EvaluateAt(*vectors, b, points);
    for (size_t i = 0; i < grad.extent(0); ++i) {
      const Real r = points[3 * i];
      CAPTURE(i);
      REQUIRE_THAT(-grad(i, 2) / (TwoPi * r),
                   WithinRel(b_values(i, 0), pcms::test::EvalTol));
      REQUIRE_THAT(grad(i, 0) / (TwoPi * r),
                   WithinRel(b_values(i, 2), pcms::test::EvalTol));
      REQUIRE_THAT(grad(i, 1), WithinAbs(0.0, pcms::test::ExactTol));
    }
    REQUIRE_THROWS_WITH(
      vectors->CreatePointEvaluator<Real>(
        vectors->Bind(pcms::test::CreateDeviceCoordinateView(
                        points, vectors->GetCoordinateSystem(), 3)
                        .coordinate_view),
        1),
      ContainsSubstring("vector quantities are not supported"));
  }

  SECTION("geqdsk: the gradient matches finite differences of the flux")
  {
    auto source = pcms::OpenFusionIOSource(FusionIOFormat::GEQDSK, GeqdskFile);
    auto space = pcms::FusionIOFunctionSpace::Create(source, 1);
    auto psi = space->CreateFunction("psi", "poloidal flux");
    // Inside a grid cell: Fusion-IO's bicubic interpolant of the geqdsk grid
    // is not differentiable across grid lines, where the finite difference
    // would straddle two cells.
    const Real r = 1.9, phi = 0.7, z = 0.263, h = 1e-5;
    const auto grad = EvaluateGradientAt(*space, psi, {r, phi, z});
    const auto values =
      EvaluateAt(*space, psi,
                 {r + h, phi, z, r - h, phi, z, r, phi, z + h, r, phi, z - h});
    // The central differences are accurate to O(h^2) relative to the gradient.
    REQUIRE_THAT(grad(0, 0),
                 WithinRel((values(0, 0) - values(1, 0)) / (2.0 * h), 1e-6));
    REQUIRE_THAT(grad(0, 2),
                 WithinRel((values(2, 0) - values(3, 0)) / (2.0 * h), 1e-6));
    REQUIRE_THAT(grad(0, 1), WithinAbs(0.0, pcms::test::ExactTol));
  }

  SECTION("M3D-C1: the flux gradient interpolates onto a grid")
  {
    auto source = pcms::OpenFusionIOSource(FusionIOFormat::M3DC1, M3DC1File);
    std::shared_ptr<const pcms::FusionIOFunctionSpace> scalars =
      pcms::FusionIOFunctionSpace::Create(source, 1);
    auto vectors = pcms::FusionIOFunctionSpace::Create(source, 3);
    const FusionIOOptions equilibrium{.part =
                                        FusionIOOptions::Part::Equilibrium};
    auto psi = scalars->CreateFunction("psi", "poloidal flux", equilibrium);
    auto b = vectors->CreateFunction("B", "magnetic field", equilibrium);

    // A grid in (R, theta, Z) inside the M3D-C1 domain.
    pcms::UniformGrid<3> grid;
    grid.bot_left = {1.5, 0.0, -0.3};
    grid.edge_length = {0.4, 1.0, 0.6};
    grid.divisions = {3, 2, 3};
    const auto system = pcms::csys::CylindricalRThetaZ::Create();
    auto value_space =
      pcms::LagrangeFunctionSpace::FromUniformGrid(grid, 1, system);
    auto vector_space =
      pcms::LagrangeFunctionSpace::FromUniformGrid(grid, 3, system);
    auto psi_t = value_space->CreateFunction<Real>("psi");
    auto grad_psi_t = vector_space->CreateFunction<Real>(
      "grad psi", pcms::values::Covector, pcms::ComponentScaling::Physical);
    auto b_t = vector_space->CreateFunction<Real>(
      "B", pcms::values::Vector, pcms::ComponentScaling::Physical);

    // The flux and its gradient share one search of the grid points.
    const auto bound = scalars->Bind(*value_space);
    pcms::Interpolator<Real>(*scalars, *value_space, bound).Apply(psi, psi_t);
    pcms::Interpolator<Real>(*pcms::Gradient(scalars), *vector_space, bound)
      .Apply(psi, grad_psi_t);
    pcms::Interpolator<Real>(*vectors, *vector_space).Apply(b, b_t);

    // Axisymmetric: B_R = -dpsi/dZ / (2 pi R), B_Z = dpsi/dR / (2 pi R).
    const auto coords = pcms::test::CopyCoordinatesToHost(
      value_space->GetLayout()->GetDOFHolderCoordinates().GetValues());
    const auto grad = grad_psi_t.GetDOFHolderDataHost().GetValues();
    const auto field = b_t.GetDOFHolderDataHost().GetValues();
    REQUIRE(coords.extent(0) == 4 * 3 * 4);
    for (size_t i = 0; i < coords.extent(0); ++i) {
      const Real r = coords(i, 0);
      CAPTURE(i, r, coords(i, 1), coords(i, 2));
      REQUIRE_THAT(-grad(i, 2) / (TwoPi * r),
                   WithinRel(field(i, 0), pcms::test::EvalTol));
      REQUIRE_THAT(grad(i, 0) / (TwoPi * r),
                   WithinRel(field(i, 2), pcms::test::EvalTol));
      REQUIRE_THAT(grad(i, 1), WithinAbs(0.0, pcms::test::ExactTol));
    }
  }

  SECTION("gradients are re-expressed in Cartesian components")
  {
    auto source = pcms::OpenFusionIOSource(FusionIOFormat::M3DC1, M3DC1File);
    std::shared_ptr<const pcms::FunctionSpace> space =
      pcms::FusionIOFunctionSpace::Create(source, 1);
    auto psi =
      std::static_pointer_cast<const pcms::FusionIOFunctionSpace>(space)
        ->CreateFunction("psi", "poloidal flux");
    const Real r = 1.7, phi = 2.2, z = 0.3;
    const auto cyl = EvaluateGradientAt(*space, psi, {r, phi, z});
    const Real expected[3] = {
      cyl(0, 0) * std::cos(phi) - cyl(0, 1) * std::sin(phi),
      cyl(0, 0) * std::sin(phi) + cyl(0, 1) * std::cos(phi), cyl(0, 2)};
    // The map and the gradient compose in either order.
    const auto to_cylindrical =
      std::make_shared<pcms::CartesianToCylindrical>();
    const std::shared_ptr<const pcms::PointEvaluatorFactory> sources[] = {
      std::make_shared<pcms::TransformedSource>(pcms::Gradient(space),
                                                to_cylindrical),
      pcms::Gradient(
        std::make_shared<pcms::TransformedSource>(space, to_cylindrical))};
    for (const auto& cartesian : sources) {
      const auto coords = pcms::test::CreateDeviceCoordinateView(
        {r * std::cos(phi), r * std::sin(phi), z},
        pcms::csys::Cartesian::Create(3), 3);
      const auto evaluator = cartesian->CreatePointEvaluator<Real>(
        cartesian->Bind(coords.coordinate_view));
      const ValueBasis basis =
        evaluator->OutputBasis(psi.GetData().GetValueBasis());
      REQUIRE(basis.variance == pcms::values::Covector);
      REQUIRE(pcms::SameCoordinateSystem(basis.system,
                                         pcms::csys::Cartesian::Create(3)));
      Kokkos::View<Real**, DeviceMemorySpace> out("out", 1, 3);
      evaluator->Evaluate(psi, pcms::ValueView<Real, DeviceMemorySpace>(
                                 basis, pcms::MakeRank2View(out)));
      const auto host =
        Kokkos::create_mirror_view_and_copy(HostMemorySpace(), out);
      for (int c = 0; c < 3; ++c) {
        CAPTURE(c);
        REQUIRE_THAT(host(0, c),
                     WithinRel(expected[c], pcms::test::EvalTol) ||
                       WithinAbs(expected[c], pcms::test::ExactTol));
      }
    }
  }
}
