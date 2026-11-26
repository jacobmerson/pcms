#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_mesh.hpp>
#include <fusionio/fusionio.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include <pcms/backends/omega_h/omega_h.hpp>
#include <pcms/field/query_points.hpp>
#include "fusion_io_source.hpp"
#include "fusion_io_space.hpp"
#include "m3dc1_native.hpp"
#include "test_utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::Quantity;
using fusionio::Real;
using fusionio::test::EvaluateAt;
using pcms::DifferentialOperator;

namespace
{

const std::string M3DC1File =
  std::string(FUSIONIO_TEST_DATA_DIR) + "/m3dc1/C1.h5";

constexpr int NumPoints = 10000;

// The corners of element e, (R, Z) for k = 0, 1, 2.
std::array<std::array<Real, 2>, 3> Corners(const m3dc1_mesh& mesh, int e)
{
  const Real local[3][2] = {{-mesh.b[e], 0.}, {mesh.a[e], 0.}, {0., mesh.c[e]}};
  std::array<std::array<Real, 2>, 3> corners{};
  for (int k = 0; k < 3; ++k) {
    Real phi = 0.;
    mesh.local_to_global(e, local[k][0], 0., local[k][1], &corners[k][0], &phi,
                         &corners[k][1]);
  }
  return corners;
}

// Random points strictly inside random elements of the mesh at random phi:
// (R, Z) and (R, phi, Z) row-major, and the element of each.
struct RandomPoints
{
  std::vector<Real> rz;
  std::vector<Real> rpz;
  std::vector<int> elements;

  [[nodiscard]] std::array<Real, 3> At(size_t i) const
  {
    return {rpz[3 * i], rpz[3 * i + 1], rpz[3 * i + 2]};
  }
};

RandomPoints InsideElements(const m3dc1_mesh& mesh, int n,
                            unsigned seed = 12345)
{
  std::mt19937 random(seed);
  std::uniform_int_distribution<int> element(0, mesh.nelms - 1);
  std::uniform_real_distribution<Real> unit(0., 1.);
  RandomPoints points;
  for (int i = 0; i < n; ++i) {
    const int e = element(random);
    Real u = unit(random), v = unit(random);
    if (u + v > 1.) {
      u = 1. - u;
      v = 1. - v;
    }
    // away from the edges, where 1.x's search tolerance could pick a
    // neighbour
    const Real shrink = 0.98;
    const Real w[3] = {shrink * (1. - u - v) + (1. - shrink) / 3.,
                       shrink * u + (1. - shrink) / 3.,
                       shrink * v + (1. - shrink) / 3.};
    const auto corners = Corners(mesh, e);
    Real r = 0., z = 0.;
    for (int k = 0; k < 3; ++k) {
      r += w[k] * corners[k][0];
      z += w[k] * corners[k][1];
    }
    const Real phi = 2. * std::numbers::pi * unit(random);
    points.rz.insert(points.rz.end(), {r, z});
    points.rpz.insert(points.rpz.end(), {r, phi, z});
    points.elements.push_back(e);
  }
  return points;
}

// Fusion-IO 1.x's options for `options`, as FusionIOFileSource sets them.
fio_option_list FusionIOOptions(fio_source& source,
                                const fusionio::Options& options)
{
  fio_option_list list;
  source.get_field_options(&list);
  if (options.timeslice) {
    list.set_option(FIO_TIMESLICE, *options.timeslice);
  }
  if (options.part) {
    list.set_option(FIO_PART, *options.part == fusionio::Part::Equilibrium
                                ? FIO_EQUILIBRIUM_ONLY
                              : *options.part == fusionio::Part::Perturbed
                                ? FIO_PERTURBED_ONLY
                                : FIO_TOTAL);
  }
  if (options.species) {
    list.set_option(FIO_SPECIES, *options.species == fusionio::Species::Electron
                                   ? FIO_ELECTRON
                                   : FIO_MAIN_ION);
  }
  if (options.linear_scale) {
    list.set_option(FIO_LINEAR_SCALE, *options.linear_scale);
  }
  if (options.phase) {
    list.set_option(FIO_PHASE, *options.phase);
  }
  return list;
}

// The largest magnitude of the column `c` of `values`, the scale of the
// absolute floor of the comparisons.
template <size_t N>
Real Scale(const std::vector<std::array<Real, N>>& values, int c)
{
  Real scale = 0.;
  for (const auto& v : values) {
    scale = std::max(scale, std::abs(v[c]));
  }
  return scale;
}

// Requires `actual` (host rows) to match `expected` within `tolerance`,
// relative or of the column's largest magnitude.
template <size_t N, typename Actual>
void RequireClose(const Actual& actual,
                  const std::vector<std::array<Real, N>>& expected,
                  Real tolerance)
{
  REQUIRE(actual.extent(0) == expected.size());
  REQUIRE(actual.extent(1) == N);
  for (int c = 0; c < static_cast<int>(N); ++c) {
    const Real scale = Scale(expected, c);
    for (size_t i = 0; i < expected.size(); ++i) {
      CAPTURE(c, i);
      REQUIRE_THAT(actual(i, c),
                   WithinRel(expected[i][c], tolerance) ||
                     WithinAbs(expected[i][c], tolerance * scale));
    }
  }
}

// Fusion-IO 1.x's values of `field` (eval) or, with `gradient`, its gradient
// as coordinate derivatives (eval_deriv) at the points.
template <size_t N>
std::vector<std::array<Real, N>> FusionIOValues(fio_field& field,
                                                const RandomPoints& points,
                                                bool gradient = false)
{
  std::vector<std::array<Real, N>> values(points.elements.size());
  for (size_t i = 0; i < values.size(); ++i) {
    int hint = points.elements[i];
    const auto x = points.At(i);
    const int status = gradient
                         ? field.eval_deriv(x.data(), values[i].data(), &hint)
                         : field.eval(x.data(), values[i].data(), &hint);
    REQUIRE(status == FIO_SUCCESS);
  }
  return values;
}

// The physical (R, phi, Z) Hessian of factor * `field` from 1.x's
// m3dc1_field::eval at the points: the coordinate second derivatives less the
// connection's terms, over the scale factors (1, R, 1).
std::vector<std::array<Real, 9>> FusionIOHessian(m3dc1_field& field,
                                                 const RandomPoints& points,
                                                 Real factor)
{
  const auto get = static_cast<m3dc1_field::m3dc1_get_op>(
    m3dc1_field::GET_VAL | m3dc1_field::GET_DVAL | m3dc1_field::GET_DDVAL |
    m3dc1_field::GET_PVAL | m3dc1_field::GET_PPVAL);
  std::vector<std::array<Real, 9>> hessians(points.elements.size());
  for (size_t i = 0; i < hessians.size(); ++i) {
    int hint = points.elements[i];
    const auto x = points.At(i);
    Real v[m3dc1_field::OP_NUM];
    REQUIRE(field.eval(x[0], x[1], x[2], get, v, &hint));
    for (auto& value : v) {
      value *= factor;
    }
    const Real r = x[0];
    const Real rphi =
      v[m3dc1_field::OP_DRP] / r - v[m3dc1_field::OP_DP] / (r * r);
    const Real phiz = v[m3dc1_field::OP_DZP] / r;
    const Real phiphi =
      v[m3dc1_field::OP_DPP] / (r * r) + v[m3dc1_field::OP_DR] / r;
    hessians[i] = {
      v[m3dc1_field::OP_DRR], rphi, v[m3dc1_field::OP_DRZ], rphi, phiphi, phiz,
      v[m3dc1_field::OP_DRZ], phiz, v[m3dc1_field::OP_DZZ]};
  }
  return hessians;
}

} // namespace

TEST_CASE("Native M3D-C1 mesh is the file's elements")
{
  auto source =
    fusionio::detail::OpenFusionIOSource(fusionio::Format::M3DC1, M3DC1File);
  const m3dc1_mesh& m3dc1 = *source->DomainMesh();
  const auto native =
    std::make_shared<fusionio::detail::M3DC1NativeMesh>(m3dc1);
  const auto geometry = native->GetGeometry();
  REQUIRE(*geometry->GetCoordinateSystem() ==
          *pcms::csys::CreateCylindricalRZ());
  Omega_h::Mesh& mesh = native->GetMesh();
  REQUIRE(mesh.dim() == 2);
  REQUIRE(mesh.nelems() == m3dc1.nelms);
  REQUIRE(native->NumElements() == m3dc1.nelms);
  // shared corners are merged: a triangulation has about half as many
  // vertices as triangles
  REQUIRE(mesh.nverts() < m3dc1.nelms);
  // and it is conforming: every interior edge has two elements
  const auto edges2elems = mesh.ask_up(1, 2);
  const Omega_h::HostRead<Omega_h::LO> offsets(edges2elems.a2ab);
  int boundary_edges = 0;
  for (int edge = 0; edge < mesh.nedges(); ++edge) {
    const int adjacent = offsets[edge + 1] - offsets[edge];
    REQUIRE((adjacent == 1 || adjacent == 2));
    boundary_edges += adjacent == 1 ? 1 : 0;
  }
  REQUIRE(boundary_edges < mesh.nedges() / 10);

  // Each element's vertices are its frame's corners, in order, up to the
  // merge tolerance (1e-4 of the smallest a + b + c; the file stores the
  // frames in single precision).
  Real smallest = m3dc1.a[0] + m3dc1.b[0] + m3dc1.c[0];
  for (int e = 0; e < m3dc1.nelms; ++e) {
    smallest = std::min(smallest, m3dc1.a[e] + m3dc1.b[e] + m3dc1.c[e]);
  }
  const Real tolerance = 1e-4 * smallest;
  const Omega_h::HostRead<Omega_h::LO> elem_verts(mesh.ask_elem_verts());
  const Omega_h::HostRead<Omega_h::Real> coords(mesh.coords());
  for (int e = 0; e < m3dc1.nelms; ++e) {
    const auto corners = Corners(m3dc1, e);
    for (int k = 0; k < 3; ++k) {
      const auto v = elem_verts[3 * e + k];
      CAPTURE(e, k);
      REQUIRE_THAT(coords[2 * v], WithinAbs(corners[k][0], tolerance));
      REQUIRE_THAT(coords[2 * v + 1], WithinAbs(corners[k][1], tolerance));
    }
  }

  // The geometry's search finds each point in the element it was drawn in.
  const auto points = InsideElements(m3dc1, 1000);
  const auto bound = fusionio::test::Bind(*geometry, points.rz);
  const auto elements = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), pcms::omega_h::LocatedElements(*bound));
  REQUIRE(elements.extent(0) == points.elements.size());
  for (size_t i = 0; i < points.elements.size(); ++i) {
    REQUIRE(elements(i) == points.elements[i]);
  }
}

TEST_CASE("Native M3D-C1 scalars match Fusion-IO 1.x")
{
  auto oracle =
    fusionio::detail::OpenFusionIOSource(fusionio::Format::M3DC1, M3DC1File);
  const auto native = fusionio::Open(fusionio::Format::M3DC1, M3DC1File,
                                     fusionio::Evaluation::Native);
  REQUIRE(*native->GetGeometry()->GetCoordinateSystem() ==
          *pcms::csys::CreateCylindricalRThetaZ());
  const auto points = InsideElements(*oracle->DomainMesh(), NumPoints);
  const auto electron = fusionio::Species::Electron;
  const auto ion = fusionio::Species::MainIon;
  const auto equilibrium = fusionio::Part::Equilibrium;
  const auto perturbed = fusionio::Part::Perturbed;
  // the equilibrium's stored scalars; then the n = 2 perturbation of time
  // slice 1 (a linear run) of the flux alone (scaled, shifted) and with the
  // equilibrium, and of the pressure
  const fusionio::Options cases[] = {
    {.part = equilibrium},
    {.part = equilibrium},
    {.part = equilibrium},
    {.part = equilibrium, .species = electron},
    {.part = equilibrium, .species = ion},
    {.part = equilibrium, .species = electron},
    {.part = equilibrium, .species = ion},
    {.part = equilibrium, .species = electron},
    {.part = equilibrium},
    {.timeslice = 1, .part = perturbed, .linear_scale = 100., .phase = 0.3},
    {.timeslice = 1, .linear_scale = 100.},
    {.timeslice = 1, .part = perturbed, .linear_scale = 100.}};
  const Quantity quantities[] = {
    Quantity::PoloidalFlux,  Quantity::PoloidalFluxNorm,
    Quantity::TotalPressure, Quantity::Pressure,
    Quantity::Density,       Quantity::Density,
    Quantity::Temperature,   Quantity::Temperature,
    Quantity::Resistivity,   Quantity::PoloidalFlux,
    Quantity::PoloidalFlux,  Quantity::TotalPressure};
  for (size_t n = 0; n < std::size(cases); ++n) {
    const auto& options = cases[n];
    const Quantity quantity = quantities[n];
    CAPTURE(n, fusionio::Name(quantity));
    const auto function = native->CreateFunction(quantity, options);
    const auto field =
      oracle->GetField(quantity, FusionIOOptions(oracle->Get(), options));
    RequireClose(EvaluateAt(function, points.rpz),
                 FusionIOValues<1>(*field, points), 1e-10);
    RequireClose(
      EvaluateAt(function, points.rpz, DifferentialOperator::Gradient),
      FusionIOValues<3>(*field, points, true), 1e-8);
  }
}

TEST_CASE("Native M3D-C1 Hessian of the poloidal flux matches Fusion-IO 1.x")
{
  auto oracle =
    fusionio::detail::OpenFusionIOSource(fusionio::Format::M3DC1, M3DC1File);
  auto& m3dc1 = dynamic_cast<m3dc1_source&>(oracle->Get());
  const auto native = fusionio::Open(fusionio::Format::M3DC1, M3DC1File,
                                     fusionio::Evaluation::Native);
  const auto points = InsideElements(*oracle->DomainMesh(), NumPoints);
  // Fusion-IO's poloidal flux is the stored psi times 2 pi B0 L0^2
  // (m3dc1_source::get_field); its fields give the coordinate derivatives.
  const Real factor = 2. * std::numbers::pi * m3dc1.B0 * m3dc1.L0 * m3dc1.L0;
  SECTION("equilibrium")
  {
    const auto psi = native->CreateFunction(
      Quantity::PoloidalFlux, {.part = fusionio::Part::Equilibrium});
    m3dc1_field* stored = m3dc1.file.load_field("psi", -1);
    REQUIRE(stored != nullptr);
    RequireClose(EvaluateAt(psi, points.rpz, DifferentialOperator::Hessian),
                 FusionIOHessian(*stored, points, factor), 1e-8);
  }
  SECTION("n = 2 perturbation")
  {
    const auto psi = native->CreateFunction(
      Quantity::PoloidalFlux,
      {.timeslice = 1, .part = fusionio::Part::Perturbed, .linear_scale = 3.});
    m3dc1_field* stored = m3dc1.file.load_field("psi", 1);
    REQUIRE(stored != nullptr);
    REQUIRE(dynamic_cast<m3dc1_complex_field*>(stored) != nullptr);
    RequireClose(EvaluateAt(psi, points.rpz, DifferentialOperator::Hessian),
                 FusionIOHessian(*stored, points, 3. * factor), 1e-8);
  }
}

TEST_CASE("Native M3D-C1 magnetic field matches Fusion-IO 1.x")
{
  auto oracle =
    fusionio::detail::OpenFusionIOSource(fusionio::Format::M3DC1, M3DC1File);
  const auto native = fusionio::Open(fusionio::Format::M3DC1, M3DC1File,
                                     fusionio::Evaluation::Native);
  const auto points = InsideElements(*oracle->DomainMesh(), NumPoints);
  // the equilibrium, the n = 2 perturbation of time slice 1 as fio_trace's
  // perturbed case takes it (with a phase), and the total
  const fusionio::Options cases[] = {{.part = fusionio::Part::Equilibrium},
                                     {.timeslice = 1,
                                      .part = fusionio::Part::Perturbed,
                                      .linear_scale = 100.,
                                      .phase = 0.7},
                                     {.timeslice = 1, .linear_scale = 100.}};
  for (const auto& options : cases) {
    CAPTURE(options.timeslice.value_or(0));
    const auto b = native->CreateFunction(Quantity::MagneticField, options);
    REQUIRE(b.GetLayout().GetNumComponents() == 3);
    const pcms::ValueBasis basis = b.GetData().GetValueBasis();
    REQUIRE(*basis.system == *pcms::csys::CreateCylindricalRThetaZ());
    REQUIRE(basis.scaling == pcms::ComponentScaling::Physical);
    REQUIRE(basis.variance == pcms::values::Vector);
    REQUIRE_FALSE(b.GetSpace().Supports(DifferentialOperator::Gradient));
    const auto field = oracle->GetField(
      Quantity::MagneticField, FusionIOOptions(oracle->Get(), options));
    RequireClose(EvaluateAt(b, points.rpz), FusionIOValues<3>(*field, points),
                 1e-10);
  }
}

TEST_CASE("Native M3D-C1 points outside the domain")
{
  auto oracle =
    fusionio::detail::OpenFusionIOSource(fusionio::Format::M3DC1, M3DC1File);
  const auto native = fusionio::Open(fusionio::Format::M3DC1, M3DC1File,
                                     fusionio::Evaluation::Native);
  const fusionio::Options equilibrium{.part = fusionio::Part::Equilibrium};
  const auto psi = native->CreateFunction(Quantity::PoloidalFlux, equilibrium);
  const auto field = oracle->GetField(
    Quantity::PoloidalFlux, FusionIOOptions(oracle->Get(), equilibrium));
  // inside points mixed with points far outside on every side
  const auto inside = InsideElements(*oracle->DomainMesh(), 4);
  std::vector<Real> rpz;
  for (const std::array<Real, 3>& x :
       {inside.At(0), std::array<Real, 3>{50.0, 0.0, 0.0},
        std::array<Real, 3>{0.05, 1.0, 0.0}, inside.At(1),
        std::array<Real, 3>{1.7, 2.0, 5.0}, std::array<Real, 3>{1.7, 3.0, -5.0},
        inside.At(2), inside.At(3)}) {
    rpz.insert(rpz.end(), x.begin(), x.end());
  }
  const std::vector<pcms::LO> outside = {1, 2, 4, 5};
  for (const pcms::LO i : outside) {
    Real value = 0.;
    REQUIRE(field->eval(&rpz[3 * static_cast<size_t>(i)], &value) ==
            FIO_OUT_OF_BOUNDS);
  }
  constexpr Real Fill = -7.0;
  const auto bound = fusionio::test::Bind(*native->GetGeometry(), rpz,
                                          {pcms::OutOfBoundsMode::FILL, Fill});
  const auto missing = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), bound->MissingPoints());
  REQUIRE(std::vector<pcms::LO>(missing.data(),
                                missing.data() + missing.extent(0)) == outside);
  const auto values = fusionio::test::EvaluateWith(
    *psi.GetSpace().CreatePointEvaluator<Real>(bound), psi);
  const auto at_inside = EvaluateAt(psi, inside.rpz);
  for (const pcms::LO i : outside) {
    REQUIRE(values(i, 0) == Fill);
  }
  const size_t inside_rows[4] = {0, 3, 6, 7};
  for (size_t k = 0; k < 4; ++k) {
    REQUIRE(values(inside_rows[k], 0) == at_inside(k, 0));
  }
  REQUIRE_THROWS_WITH(fusionio::test::Bind(*native->GetGeometry(), rpz),
                      ContainsSubstring("outside the source's domain"));

  // QueryPoints rebinds in place and evaluates the moved points.
  const auto& geometry = native->GetGeometry();
  pcms::QueryPoints query(geometry);
  const auto first = fusionio::test::CreateDeviceCoordinateView(
    inside.rpz, geometry->GetCoordinateSystem(), 3);
  query.Locate(first.coordinate_view);
  const auto* storage = query.Bound().get();
  const auto moved_points = InsideElements(*oracle->DomainMesh(), 4, 54321);
  const auto moved = fusionio::test::CreateDeviceCoordinateView(
    moved_points.rpz, geometry->GetCoordinateSystem(), 3);
  query.Locate(moved.coordinate_view);
  REQUIRE(query.Bound().get() == storage);
  Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", 4, 1);
  query.Evaluate(psi, fusionio::test::TagLike(psi, out));
  const auto rebound =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), out);
  const auto fresh = EvaluateAt(psi, moved_points.rpz);
  for (size_t k = 0; k < 4; ++k) {
    REQUIRE(rebound(k, 0) == fresh(k, 0));
  }
}

TEST_CASE("Native M3D-C1 sources refuse what they do not evaluate")
{
  REQUIRE_THROWS_WITH(fusionio::Open(fusionio::Format::GEQDSK,
                                     std::string(FUSIONIO_TEST_DATA_DIR) +
                                       "/geqdsk/g158115.04701",
                                     fusionio::Evaluation::Native),
                      ContainsSubstring("only M3D-C1"));
  const auto native = fusionio::Open(fusionio::Format::M3DC1, M3DC1File,
                                     fusionio::Evaluation::Native);
  REQUIRE_THROWS_WITH(
    native->CreateFunction(Quantity::Pressure,
                           {.part = fusionio::Part::Equilibrium,
                            .species = fusionio::Species::MainIon}),
    ContainsSubstring("not evaluated natively"));
  REQUIRE_THROWS_WITH(native->CreateFunction(Quantity::CurrentDensity),
                      ContainsSubstring("not evaluated natively"));
  REQUIRE_THROWS_WITH(native->CreateFunction(Quantity::Density),
                      ContainsSubstring("needs the species option"));
  // the wrapped source's description
  REQUIRE(native->Times().size() == 2);
  REQUIRE_THAT(
    native->GetGeometry()->GetCoordinateSystem()->PeriodicAxes().front().length,
    WithinRel(2 * std::numbers::pi, 1e-14));
}
