#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_mesh.hpp>
#include <fusionio/fusionio.hpp>
#include <pcms/field/coordinate_systems/cylindrical.hpp>
#include <pcms/field/evaluator/omega_h_geometry.hpp>
#include "fusion_io_source.hpp"
#include "fusion_io_space.hpp"
#include "m3dc1_native.hpp"
#include "test_utils.hpp"
#include <algorithm>
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

// Random points strictly inside random elements of the mesh, (R, Z)
// row-major, and the element of each.
struct RandomPoints
{
  std::vector<Real> rz;
  std::vector<int> elements;
};

RandomPoints InsideElements(const m3dc1_mesh& mesh, int n)
{
  std::mt19937 random(12345);
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
    points.rz.insert(points.rz.end(), {r, z});
    points.elements.push_back(e);
  }
  return points;
}

// (R, phi, Z) of point i of `rz`.
std::array<Real, 3> RPhiZ(const std::vector<Real>& rz, size_t i)
{
  return {rz[2 * i], 0., rz[2 * i + 1]};
}

// Fusion-IO 1.x's options for `options` (as FusionIOFileSource sets them).
fio_option_list FusionIOOptions(fio_source& source,
                                const fusionio::Options& options)
{
  fio_option_list list;
  source.get_field_options(&list);
  list.set_option(FIO_PART, FIO_EQUILIBRIUM_ONLY);
  if (options.species) {
    list.set_option(FIO_SPECIES, *options.species == fusionio::Species::Electron
                                   ? FIO_ELECTRON
                                   : FIO_MAIN_ION);
  }
  return list;
}

// The largest magnitude of the column `c` of `values`, the scale of the
// absolute floor of the comparisons.
Real Scale(const std::vector<std::array<Real, 3>>& values, int c)
{
  Real scale = 0.;
  for (const auto& v : values) {
    scale = std::max(scale, std::abs(v[c]));
  }
  return scale;
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
  REQUIRE(pcms::SameCoordinateSystem(geometry->GetCoordinateSystem(),
                                     pcms::csys::CylindricalRZ::Create()));
  Omega_h::Mesh& mesh = pcms::GetMesh(*geometry);
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
  const auto& binding = dynamic_cast<const pcms::OmegaHMeshBinding&>(*bound);
  const auto elements = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), binding.ElementIds());
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
  const auto points = InsideElements(*oracle->DomainMesh(), NumPoints);
  const auto electron = fusionio::Species::Electron;
  const auto ion = fusionio::Species::MainIon;
  struct Case
  {
    Quantity quantity;
    std::optional<fusionio::Species> species;
  };
  for (const auto& [quantity, species] :
       {Case{Quantity::PoloidalFlux, {}}, Case{Quantity::PoloidalFluxNorm, {}},
        Case{Quantity::TotalPressure, {}}, Case{Quantity::Pressure, electron},
        Case{Quantity::Density, ion}, Case{Quantity::Density, electron},
        Case{Quantity::Temperature, ion}, Case{Quantity::Temperature, electron},
        Case{Quantity::Resistivity, {}}}) {
    CAPTURE(fusionio::Name(quantity), species.has_value());
    const fusionio::Options options{.part = fusionio::Part::Equilibrium,
                                    .species = species};
    const auto function = native->CreateFunction(quantity, options);
    const auto field =
      oracle->GetField(quantity, FusionIOOptions(oracle->Get(), options));
    std::vector<std::array<Real, 3>> values(NumPoints), gradients(NumPoints);
    for (size_t i = 0; i < NumPoints; ++i) {
      int hint = points.elements[i];
      const auto x = RPhiZ(points.rz, i);
      REQUIRE(field->eval(x.data(), values[i].data(), &hint) == FIO_SUCCESS);
      REQUIRE(field->eval_deriv(x.data(), gradients[i].data(), &hint) ==
              FIO_SUCCESS);
    }
    const auto value = EvaluateAt(function, points.rz);
    const auto gradient =
      EvaluateAt(function, points.rz, DifferentialOperator::Gradient);
    REQUIRE(gradient.extent(1) == 2);
    const Real scale = Scale(values, 0);
    const Real scale_r = Scale(gradients, 0);
    const Real scale_z = Scale(gradients, 2);
    for (size_t i = 0; i < NumPoints; ++i) {
      CAPTURE(i, points.rz[2 * i], points.rz[2 * i + 1]);
      REQUIRE_THAT(value(i, 0), WithinRel(values[i][0], 1e-10) ||
                                  WithinAbs(values[i][0], 1e-10 * scale));
      REQUIRE_THAT(gradient(i, 0),
                   WithinRel(gradients[i][0], 1e-8) ||
                     WithinAbs(gradients[i][0], 1e-8 * scale_r));
      REQUIRE_THAT(gradient(i, 1),
                   WithinRel(gradients[i][2], 1e-8) ||
                     WithinAbs(gradients[i][2], 1e-8 * scale_z));
    }
  }
}

TEST_CASE("Native M3D-C1 Hessian of the poloidal flux matches Fusion-IO 1.x")
{
  auto oracle =
    fusionio::detail::OpenFusionIOSource(fusionio::Format::M3DC1, M3DC1File);
  auto& m3dc1 = dynamic_cast<m3dc1_source&>(oracle->Get());
  const auto native = fusionio::Open(fusionio::Format::M3DC1, M3DC1File,
                                     fusionio::Evaluation::Native);
  const auto psi = native->CreateFunction(
    Quantity::PoloidalFlux, {.part = fusionio::Part::Equilibrium});
  const auto points = InsideElements(*oracle->DomainMesh(), NumPoints);
  const auto hessian =
    EvaluateAt(psi, points.rz, DifferentialOperator::Hessian);
  REQUIRE(hessian.extent(1) == 4);
  // Fusion-IO's poloidal flux is the stored psi times 2 pi B0 L0^2
  // (m3dc1_source::get_field); its fields give the second derivatives.
  m3dc1_field* stored = m3dc1.file.load_field("psi", -1);
  REQUIRE(stored != nullptr);
  const Real factor = 2. * std::numbers::pi * m3dc1.B0 * m3dc1.L0 * m3dc1.L0;
  std::vector<std::array<Real, 3>> expected(NumPoints);
  for (size_t i = 0; i < NumPoints; ++i) {
    int hint = points.elements[i];
    Real val[m3dc1_field::OP_NUM];
    REQUIRE(stored->eval(points.rz[2 * i], 0., points.rz[2 * i + 1],
                         m3dc1_field::GET_DDVAL, val, &hint));
    expected[i] = {factor * val[m3dc1_field::OP_DRR],
                   factor * val[m3dc1_field::OP_DRZ],
                   factor * val[m3dc1_field::OP_DZZ]};
  }
  const int columns[4] = {0, 1, 1, 2};
  for (int c = 0; c < 4; ++c) {
    const Real scale = Scale(expected, columns[c]);
    for (size_t i = 0; i < NumPoints; ++i) {
      CAPTURE(c, i);
      const Real e = expected[i][columns[c]];
      REQUIRE_THAT(hessian(i, c),
                   WithinRel(e, 1e-8) || WithinAbs(e, 1e-8 * scale));
    }
  }
}

TEST_CASE("Native M3D-C1 equilibrium magnetic field matches Fusion-IO 1.x")
{
  auto oracle =
    fusionio::detail::OpenFusionIOSource(fusionio::Format::M3DC1, M3DC1File);
  const auto native = fusionio::Open(fusionio::Format::M3DC1, M3DC1File,
                                     fusionio::Evaluation::Native);
  const fusionio::Options equilibrium{.part = fusionio::Part::Equilibrium};
  const auto b = native->CreateFunction(Quantity::MagneticField, equilibrium);
  REQUIRE(b.GetLayout().GetNumComponents() == 3);
  REQUIRE(pcms::SameValueBasis(
    b.GetData().GetValueBasis(),
    pcms::ValueBasis{pcms::csys::CylindricalRThetaZ::Create(),
                     pcms::ComponentScaling::Physical, pcms::values::Vector}));
  REQUIRE_FALSE(b.GetSpace().Supports(DifferentialOperator::Gradient));
  const auto field = oracle->GetField(
    Quantity::MagneticField, FusionIOOptions(oracle->Get(), equilibrium));
  const auto points = InsideElements(*oracle->DomainMesh(), NumPoints);
  std::vector<std::array<Real, 3>> expected(NumPoints);
  for (size_t i = 0; i < NumPoints; ++i) {
    int hint = points.elements[i];
    const auto x = RPhiZ(points.rz, i);
    REQUIRE(field->eval(x.data(), expected[i].data(), &hint) == FIO_SUCCESS);
  }
  const auto values = EvaluateAt(b, points.rz);
  for (int c = 0; c < 3; ++c) {
    const Real scale = Scale(expected, c);
    REQUIRE(scale > 0.);
    for (size_t i = 0; i < NumPoints; ++i) {
      CAPTURE(c, i, points.rz[2 * i], points.rz[2 * i + 1]);
      REQUIRE_THAT(values(i, c), WithinRel(expected[i][c], 1e-10) ||
                                   WithinAbs(expected[i][c], 1e-10 * scale));
    }
  }
  // the time slices' fields are not axisymmetric
  REQUIRE_THROWS_WITH(
    native->CreateFunction(Quantity::MagneticField,
                           {.timeslice = 1, .part = fusionio::Part::Perturbed}),
    ContainsSubstring("not axisymmetric"));
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
  const std::vector<Real> rz = {
    inside.rz[0], inside.rz[1], 50.0,         0.0,         0.05, 0.0,
    inside.rz[2], inside.rz[3], 1.7,          5.0,         1.7,  -5.0,
    inside.rz[4], inside.rz[5], inside.rz[6], inside.rz[7]};
  const std::vector<pcms::LO> outside = {1, 2, 4, 5};
  for (const pcms::LO i : outside) {
    const auto x = RPhiZ(rz, static_cast<size_t>(i));
    Real value = 0.;
    REQUIRE(field->eval(x.data(), &value) == FIO_OUT_OF_BOUNDS);
  }
  constexpr Real Fill = -7.0;
  const auto bound = fusionio::test::Bind(*native->GetGeometry(), rz,
                                          {pcms::OutOfBoundsMode::FILL, Fill});
  const auto missing = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), bound->MissingPoints());
  REQUIRE(std::vector<pcms::LO>(missing.data(),
                                missing.data() + missing.extent(0)) == outside);
  const auto values = fusionio::test::EvaluateWith(
    *psi.GetSpace().CreatePointEvaluator<Real>(bound), psi);
  const auto at_inside = EvaluateAt(psi, inside.rz);
  for (const pcms::LO i : outside) {
    REQUIRE(values(i, 0) == Fill);
  }
  const size_t inside_rows[4] = {0, 3, 6, 7};
  for (size_t k = 0; k < 4; ++k) {
    REQUIRE(values(inside_rows[k], 0) == at_inside(k, 0));
  }
  REQUIRE_THROWS_WITH(fusionio::test::Bind(*native->GetGeometry(), rz),
                      ContainsSubstring("outside the source's domain"));
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
  // time slice 1 holds an n = 2 perturbation
  REQUIRE_THROWS_WITH(
    native->CreateFunction(Quantity::PoloidalFlux,
                           {.timeslice = 1, .part = fusionio::Part::Perturbed}),
    ContainsSubstring("not axisymmetric"));
  REQUIRE_THROWS_WITH(
    native->CreateFunction(Quantity::Pressure,
                           {.part = fusionio::Part::Equilibrium,
                            .species = fusionio::Species::MainIon}),
    ContainsSubstring("not evaluated natively"));
  REQUIRE_THROWS_WITH(native->CreateFunction(Quantity::Density),
                      ContainsSubstring("needs the species option"));
  // the wrapped source's description
  REQUIRE(native->Times().size() == 2);
  REQUIRE_THAT(native->Period(), WithinRel(2 * std::numbers::pi, 1e-14));
}
