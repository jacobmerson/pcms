#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>

#include <pcms/transfer/mass_matrix_type.hpp>
#include <pcms/transfer/omega_h_conservative_projection.hpp>
#include <pcms/transfer/omega_h_form_integrator_utils.hpp>
#include <pcms/utility/arrays.h>
#include "field_test_utils.h"

#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace
{

// Tets taken verbatim from the PUMI-Tally wistell-D mesh. Clipping each one
// against itself with r3d gives signed distances of ~1e-13 on every face plane,
// and r3d then interpolates new vertices at O(1) fractions along the edges
// between two on-plane vertices. The resulting polytope has the right volume
// but a folded face graph whose triangles come out in both orientations, so a
// star decomposition that takes |volume| per piece over-counts (by 28.6%,
// 1.7% and 9.0% respectively). Any element sharing a face plane with a source
// element can hit this; the same-mesh case hits it for every element.
const std::vector<double> kFoldingTetCoords = {
  30.198030285537243,
  -936.1389388516545,
  483.16848456836186,
  20.132020190358162,
  -979.0909772444453,
  485.85648258043625,
  0.0,
  -966.3369691371918,
  483.16848456836186,
  15.099015142768621,
  -951.2379539944232,
  468.06946942559324,

  -452.97045428305864,
  392.57439371198416,
  241.58424228406392,
  -457.87167295326776,
  423.1168047352564,
  234.21738729666106,
  -482.6122426477242,
  412.69480663896707,
  261.65815839972544,
  -460.4863470953233,
  422.81883403751397,
  272.4806339440611,

  -620.6927499466186,
  1017.0028114430277,
  82.31454644833995,
  -634.1586359962821,
  1056.9310599938033,
  90.59409085637769,
  -637.7105881441516,
  1022.8656314424136,
  102.40466297536494,
  -603.9377425704466,
  1022.2805800406807,
  87.57395423730232,
};

double SignedTetVolume(const double* c)
{
  const double b[3] = {c[3] - c[0], c[4] - c[1], c[5] - c[2]};
  const double d[3] = {c[6] - c[0], c[7] - c[1], c[8] - c[2]};
  const double e[3] = {c[9] - c[0], c[10] - c[1], c[11] - c[2]};
  return (b[0] * (d[1] * e[2] - d[2] * e[1]) -
          b[1] * (d[0] * e[2] - d[2] * e[0]) +
          b[2] * (d[0] * e[1] - d[1] * e[0])) /
         6.0;
}

// Random, positively oriented, non-sliver tets at the coordinate scale of a
// tokamak/stellarator mesh (~1e3), so shared-plane roundoff is representative.
std::vector<double> RandomTetCoords(int ntets, unsigned seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> base(-1900.0, 1900.0);
  std::uniform_real_distribution<double> offset(-50.0, 50.0);
  std::vector<double> coords;
  coords.reserve(12 * ntets);
  while (static_cast<int>(coords.size()) < 12 * ntets) {
    double c[12];
    for (int d = 0; d < 3; ++d) {
      c[d] = base(gen);
    }
    for (int v = 1; v < 4; ++v) {
      for (int d = 0; d < 3; ++d) {
        c[3 * v + d] = c[d] + offset(gen);
      }
    }
    const double vol = SignedTetVolume(c);
    if (std::fabs(vol) < 1e3) {
      continue;
    }
    if (vol < 0.0) {
      for (int d = 0; d < 3; ++d) {
        std::swap(c[6 + d], c[9 + d]);
      }
    }
    coords.insert(coords.end(), c, c + 12);
  }
  return coords;
}

// One mesh holding `coords.size() / 12` mutually disjoint tets.
Omega_h::Mesh BuildDisjointTets(Omega_h::Library& lib,
                                const std::vector<double>& coords)
{
  const int ntets = static_cast<int>(coords.size()) / 12;
  Omega_h::HostWrite<Omega_h::LO> ev2v(4 * ntets);
  for (int i = 0; i < 4 * ntets; ++i) {
    ev2v[i] = i;
  }
  Omega_h::HostWrite<Omega_h::Real> x(static_cast<Omega_h::LO>(coords.size()));
  for (std::size_t i = 0; i < coords.size(); ++i) {
    x[static_cast<Omega_h::LO>(i)] = coords[i];
  }
  Omega_h::Mesh mesh(&lib);
  Omega_h::build_from_elems_and_coords(&mesh, OMEGA_H_SIMPLEX, 3,
                                       Omega_h::LOs(ev2v.write()),
                                       Omega_h::Reals(x.write()));
  pcms::test::AddDefaultClassification(mesh);
  return mesh;
}

// Unit cube tessellation moved to non-dyadic coordinates of magnitude ~1e3.
// On the unit cube every on-plane distance is exactly zero and r3d never
// folds; this is what a real mesh looks like to the clipper.
Omega_h::Mesh BuildScaledBox(Omega_h::Library& lib, int n)
{
  auto mesh = pcms::test::BuildUnitCube(lib, n);
  const auto coords = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  Omega_h::HostWrite<Omega_h::Real> scaled(coords.size());
  constexpr double scale = 937.31;
  constexpr std::array<double, 3> shift = {-1894.78, 211.7, -598.52};
  for (Omega_h::LO v = 0; v < mesh.nverts(); ++v) {
    for (int d = 0; d < 3; ++d) {
      scaled[3 * v + d] = coords[3 * v + d] * scale + shift[d];
    }
  }
  mesh.set_coords(Omega_h::Reals(scaled.write()));
  return mesh;
}

// Sum of the sub-simplex measures handed to the integrand for target element
// e when the source mesh is the target mesh itself and the intersection map is
// the identity. Exactly |e| for a correct decomposition.
Omega_h::HostRead<Omega_h::Real> SelfDecompositionVolumes(Omega_h::Mesh& mesh)
{
  const int nelems = mesh.nelems();
  const auto coords = mesh.coords();
  const auto e2n = mesh.ask_down(3, Omega_h::VERT).ab2b;
  Omega_h::Write<Omega_h::LO> offsets(nelems + 1);
  Omega_h::Write<Omega_h::LO> indices(nelems);
  Omega_h::parallel_for(nelems + 1, OMEGA_H_LAMBDA(int i) { offsets[i] = i; });
  Omega_h::parallel_for(nelems, OMEGA_H_LAMBDA(int i) { indices[i] = i; });
  const pcms::IntersectionResults self{Omega_h::LOs(offsets),
                                       Omega_h::LOs(indices)};

  Omega_h::Write<Omega_h::Real> sums(nelems, 0.0);
  Omega_h::parallel_for(
    nelems, OMEGA_H_LAMBDA(int e) {
      Omega_h::Real acc = 0.0;
      pcms::detail::ForEachIntersectionSubsimplex<3>(
        e, self, coords, coords, e2n, e2n,
        [&](const Omega_h::Few<Omega_h::Vector<3>, 4>&, int,
            Omega_h::Real measure) { acc += measure; });
      sums[e] = acc;
    });
  return Omega_h::HostRead<Omega_h::Real>(sums);
}

void RequireSelfDecompositionExact(Omega_h::Mesh& mesh)
{
  const auto sums = SelfDecompositionVolumes(mesh);
  const auto sizes =
    Omega_h::HostRead<Omega_h::Real>(Omega_h::measure_elements_real(&mesh));
  for (Omega_h::LO e = 0; e < mesh.nelems(); ++e) {
    const double size = std::fabs(sizes[e]);
    REQUIRE(sums[e] == Catch::Approx(size).epsilon(1e-12));
  }
}

// The lumped P1 mass matrix is diagonal with M_ii = sum_{e in i} |e|/4 and the
// exact P0 load vector is b_i = sum_{e in i} |e|/4 f_e, so the lumped
// projection of a P0 field onto P1 on the same mesh is the volume-weighted
// average of the incident element values. Nothing here is approximate; the
// operator must reproduce it to roundoff.
std::vector<double> ElementNodalAverage(Omega_h::Mesh& mesh,
                                        const std::vector<double>& elem_values)
{
  const auto sizes =
    Omega_h::HostRead<Omega_h::Real>(Omega_h::measure_elements_real(&mesh));
  const auto e2v = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());
  std::vector<double> numerator(mesh.nverts(), 0.0);
  std::vector<double> denominator(mesh.nverts(), 0.0);
  for (Omega_h::LO e = 0; e < mesh.nelems(); ++e) {
    const double w = std::fabs(sizes[e]) / 4.0;
    for (int k = 0; k < 4; ++k) {
      const auto v = e2v[4 * e + k];
      numerator[v] += w * elem_values[e];
      denominator[v] += w;
    }
  }
  for (Omega_h::LO v = 0; v < mesh.nverts(); ++v) {
    numerator[v] /= denominator[v];
  }
  return numerator;
}

template <typename View>
std::vector<double> ToVector(const View& values)
{
  std::vector<double> out(values.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = values[i];
  }
  return out;
}

// Zeroth and first moments (integrals of 1, x, y, z times the field) of a P0
// element field and of a P1 vertex field, both exact on linear tets.
std::array<double, 4> P0Moments(Omega_h::Mesh& mesh,
                                const std::vector<double>& f)
{
  const auto sizes =
    Omega_h::HostRead<Omega_h::Real>(Omega_h::measure_elements_real(&mesh));
  const auto e2v = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());
  const auto x = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  std::array<double, 4> m = {0.0, 0.0, 0.0, 0.0};
  for (Omega_h::LO e = 0; e < mesh.nelems(); ++e) {
    const double vol = std::fabs(sizes[e]);
    m[0] += vol * f[e];
    for (int d = 0; d < 3; ++d) {
      double centroid = 0.0;
      for (int k = 0; k < 4; ++k) {
        centroid += x[3 * e2v[4 * e + k] + d];
      }
      m[1 + d] += vol * f[e] * centroid / 4.0;
    }
  }
  return m;
}

std::array<double, 4> P1Moments(Omega_h::Mesh& mesh,
                                const std::vector<double>& u)
{
  const auto sizes =
    Omega_h::HostRead<Omega_h::Real>(Omega_h::measure_elements_real(&mesh));
  const auto e2v = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());
  const auto x = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  std::array<double, 4> m = {0.0, 0.0, 0.0, 0.0};
  for (Omega_h::LO e = 0; e < mesh.nelems(); ++e) {
    const double vol = std::fabs(sizes[e]);
    double sum_u = 0.0;
    for (int k = 0; k < 4; ++k) {
      sum_u += u[e2v[4 * e + k]];
    }
    m[0] += vol * sum_u / 4.0;
    // int_T phi_i phi_j = |T| (1 + delta_ij) / 20 gives
    // int_T x u_h = |T| / 20 * ((sum_i u_i)(sum_j x_j) + sum_i u_i x_i).
    for (int d = 0; d < 3; ++d) {
      double sum_x = 0.0;
      double sum_ux = 0.0;
      for (int k = 0; k < 4; ++k) {
        const double xk = x[3 * e2v[4 * e + k] + d];
        sum_x += xk;
        sum_ux += u[e2v[4 * e + k]] * xk;
      }
      m[1 + d] += vol / 20.0 * (sum_u * sum_x + sum_ux);
    }
  }
  return m;
}

// A P0 field that varies from element to element at the ~10% level, so a
// misattributed or double-counted piece of any element is visible.
struct SourceFunction
{
  KOKKOS_INLINE_FUNCTION pcms::Real operator()(pcms::Real x, pcms::Real y,
                                               pcms::Real z) const
  {
    return 10.0 + Kokkos::sin(0.011 * x) + 0.5 * Kokkos::cos(0.017 * y) +
           0.001 * z;
  }
};

std::vector<double> ProjectSameMesh(Omega_h::Mesh& mesh,
                                    pcms::MassMatrixType mass_type,
                                    std::vector<double>& source_values)
{
  auto source_space = pcms::test::MakeP0Space(mesh);
  auto target_space = pcms::test::MakeP1Space(mesh);
  auto source = source_space->CreateFunction<pcms::Real>();
  auto target = target_space->CreateFunction<pcms::Real>();
  pcms::test::SetField(source, SourceFunction{});

  pcms::OmegaHConservativeProjection projection(*source_space, *target_space,
                                                mass_type);
  projection.Apply(source, target);

  source_values =
    ToVector(pcms::FlattenToRank1View(source.GetDOFHolderDataHost()));
  return ToVector(pcms::FlattenToRank1View(target.GetDOFHolderDataHost()));
}

void RequireLumpedEqualsNodalAverage(Omega_h::Mesh& mesh)
{
  std::vector<double> f;
  const auto u = ProjectSameMesh(mesh, pcms::MassMatrixType::Lumped, f);
  REQUIRE(static_cast<Omega_h::LO>(u.size()) == mesh.nverts());
  const auto expected = ElementNodalAverage(mesh, f);
  for (Omega_h::LO v = 0; v < mesh.nverts(); ++v) {
    REQUIRE(u[v] == Catch::Approx(expected[v]).epsilon(1e-12));
  }
  REQUIRE(P1Moments(mesh, u)[0] ==
          Catch::Approx(P0Moments(mesh, f)[0]).epsilon(1e-12));
}

// The polytope r3d produced when clipping tet 38125 above against itself in
// an optimized build, captured verbatim (positions to 1e-6, graph exact). The
// four original vertices appear three or four times each and three new
// vertices sit at 1/2 and 2/3 along edges; two of the faces the walk emits are
// mirror images that must cancel. Whether a given build reproduces this fold
// from the coordinates depends on the compiler's floating-point contraction,
// so the polytope is pinned here rather than regenerated.
r3d::Polytope<3> FoldedSelfClipPolytope()
{
  constexpr int nverts = 14;
  constexpr double pos[nverts][3] = {
    {0.000000, -966.336969, 483.168485},  {20.132020, -979.090977, 485.856483},
    {7.549508, -958.787462, 475.618977},  {0.000000, -966.336969, 483.168485},
    {30.198030, -936.138939, 483.168485}, {15.099015, -951.237954, 468.069469},
    {22.648523, -943.688446, 475.618977}, {30.198030, -936.138939, 483.168485},
    {0.000000, -966.336969, 483.168485},  {20.132020, -979.090977, 485.856483},
    {20.132020, -979.090977, 485.856483}, {13.421347, -974.839641, 484.960483},
    {20.132020, -979.090977, 485.856483}, {30.198030, -936.138939, 483.168485},
  };
  constexpr int pnbrs[nverts][3] = {
    {3, 2, 8},  {4, 9, 10}, {0, 5, 11},  {0, 4, 6},   {1, 7, 3},
    {2, 6, 12}, {3, 7, 5},  {4, 13, 6},  {0, 11, 9},  {1, 8, 10},
    {1, 9, 13}, {2, 12, 8}, {5, 13, 11}, {7, 10, 12},
  };
  r3d::Polytope<3> poly;
  poly.nverts = nverts;
  for (int v = 0; v < nverts; ++v) {
    for (int d = 0; d < 3; ++d) {
      poly.verts[v].pos[d] = pos[v][d];
      poly.verts[v].pnbrs[d] = pnbrs[v][d];
    }
  }
  return poly;
}

double StarDecompositionVolume(const r3d::Polytope<3>& poly)
{
  const double eps_vol =
    PCMS_INTERSECTION_ABS_TOL +
    PCMS_INTERSECTION_REL_TOL * std::fabs(r3d::measure(poly));
  double sum = 0.0;
  pcms::detail::ForEachPolytopeStarTet(
    poly, eps_vol,
    [&](const Omega_h::Few<Omega_h::Vector<3>, 4>&, Omega_h::Real measure) {
      sum += measure;
    });
  return sum;
}

} // namespace

TEST_CASE("star decomposition of a folded r3d polytope matches its volume",
          "[transfer][mesh_intersection][3d]")
{
  SECTION("clean tetrahedron: four pieces, each a quarter of the volume")
  {
    r3d::Few<r3d::Vector<3>, 4> verts;
    for (int v = 0; v < 4; ++v) {
      for (int d = 0; d < 3; ++d) {
        verts[v][d] = kFoldingTetCoords[3 * v + d];
      }
    }
    r3d::Polytope<3> poly;
    r3d::init(poly, verts);
    const double volume = std::fabs(r3d::measure(poly));
    int pieces = 0;
    pcms::detail::ForEachPolytopeStarTet(
      poly, 0.0,
      [&](const Omega_h::Few<Omega_h::Vector<3>, 4>&, Omega_h::Real measure) {
        REQUIRE(measure == Catch::Approx(volume / 4.0).epsilon(1e-12));
        ++pieces;
      });
    REQUIRE(pieces == 4);
  }

  SECTION("folded self-clip polytope")
  {
    const auto poly = FoldedSelfClipPolytope();
    const double volume = std::fabs(r3d::measure(poly));
    // Volume of the original tet (positions were rounded to 1e-6): r3d's own
    // signed integration gets it right despite the fold.
    REQUIRE(volume == Catch::Approx(2499.121742).epsilon(1e-6));
    REQUIRE(StarDecompositionVolume(poly) ==
            Catch::Approx(volume).epsilon(1e-12));
  }
}

TEST_CASE("self-clip star decomposition sums to the element volume",
          "[transfer][mesh_intersection][3d]")
{
  Omega_h::Library lib;

  SECTION("tets whose r3d self-clip is known to fold")
  {
    auto mesh = BuildDisjointTets(lib, kFoldingTetCoords);
    REQUIRE(mesh.nelems() == 3);
    RequireSelfDecompositionExact(mesh);
  }

  SECTION("random tets at realistic coordinate scale")
  {
    auto mesh = BuildDisjointTets(lib, RandomTetCoords(400, 20260902u));
    RequireSelfDecompositionExact(mesh);
  }

  SECTION("scaled box mesh")
  {
    auto mesh = BuildScaledBox(lib, 5);
    RequireSelfDecompositionExact(mesh);
  }
}

TEST_CASE("same-mesh P0->P1 lumped projection equals the element-nodal average",
          "[transfer][mesh_intersection][3d]")
{
  Omega_h::Library lib;

  SECTION("tets whose r3d self-clip is known to fold")
  {
    auto mesh = BuildDisjointTets(lib, kFoldingTetCoords);
    RequireLumpedEqualsNodalAverage(mesh);
  }

  SECTION("scaled box mesh")
  {
    auto mesh = BuildScaledBox(lib, 5);
    RequireLumpedEqualsNodalAverage(mesh);
  }
}

TEST_CASE("same-mesh P0->P1 consistent projection preserves zeroth and first "
          "moments",
          "[transfer][mesh_intersection][3d]")
{
  // Galerkin orthogonality with v = 1, x, y, z in P1: the consistent L2
  // projection matches every moment the target space can represent.
  Omega_h::Library lib;
  auto mesh = BuildScaledBox(lib, 5);
  std::vector<double> f;
  const auto u = ProjectSameMesh(mesh, pcms::MassMatrixType::Consistent, f);
  const auto m0 = P0Moments(mesh, f);
  const auto m1 = P1Moments(mesh, u);
  for (int k = 0; k < 4; ++k) {
    REQUIRE(m1[k] == Catch::Approx(m0[k]).epsilon(1e-9));
  }
}
