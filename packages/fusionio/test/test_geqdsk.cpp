#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <fusionio/geqdsk.hpp>
#include <pcms/field/query_points.hpp>
#include "test_utils.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::Quantity;
using fusionio::Real;
using fusionio::test::EvaluateAt;

namespace
{

// A geqdsk equilibrium whose grid values are cubic polynomials, which the
// cubic splines reproduce exactly: psi(R, Z) = a x^2 + b Z^2 + c x^3 with
// x = R - 1.5, on R in [1, 2], Z in [-1, 1], from the axis (psi = 0 at
// (1.5, 0)) to the boundary at psi = 0.1. A `sign` of -1 stores the flux as
// 0.1 - psi, which decreases outward.
struct Synthetic
{
  Real sign;

  Real Psi(Real r, Real z) const
  {
    const Real x = r - 1.5;
    const Real psi = 0.3 * x * x + 0.2 * z * z + 0.1 * x * x * x;
    return sign > 0 ? psi : 0.1 - psi;
  }
  Real PsiR(Real r) const
  {
    const Real x = r - 1.5;
    return sign * (0.6 * x + 0.3 * x * x);
  }
  Real PsiZ(Real z) const { return sign * 0.4 * z; }
  // the profiles as polynomials in the file's flux
  static Real F(Real psi) { return 2 + 0.5 * psi + 0.3 * psi * psi; }
  static Real P(Real psi) { return 1000 + 4000 * psi - 9000 * psi * psi; }
  static Real Q(Real psi) { return 1 + 2 * psi - 3 * psi * psi * psi; }

  void Write(const std::string& path) const
  {
    constexpr int nw = 33, nh = 41;
    const Real simag = sign > 0 ? 0 : 0.1, sibry = sign > 0 ? 0.1 : 0;
    std::ofstream file(path);
    file << std::setprecision(17);
    file << "SYNTHETIC 01/01/2026 #0 0ms 3 " << nw << " " << nh << "\n";
    file << 1.0 << " " << 2.0 << " " << 1.5 << " " << 1.0 << " " << 0.0 << "\n";
    file << 1.5 << " " << 0.0 << " " << simag << " " << sibry << " " << 2.0
         << "\n";
    file << "1e6 " << simag << " 0 1.5 0\n0 0 " << sibry << " 0 0\n";
    const auto profile = [&](Real (*f)(Real)) {
      for (int i = 0; i < nw; ++i) {
        file << f(simag + (sibry - simag) * i / (nw - 1)) << "\n";
      }
    };
    profile(F);
    profile(P);
    profile([](Real) { return Real(0); }); // ffprim
    profile([](Real) { return Real(0); }); // pprime
    for (int j = 0; j < nh; ++j) {
      for (int i = 0; i < nw; ++i) {
        file << Psi(1.0 + i / Real(nw - 1), -1.0 + 2.0 * j / (nh - 1)) << "\n";
      }
    }
    profile(Q);
    file << "5 4\n";
    file << "1.2 0 1.5 0.5 1.8 0 1.5 -0.5 1.2 0\n";
    file << "1.1 -0.9 1.9 -0.9 1.9 0.9 1.1 0.9\n";
  }
};

// The file of a Synthetic, removed with it.
class SyntheticFile
{
public:
  explicit SyntheticFile(Real sign) : synthetic_{sign}
  {
    synthetic_.Write(path_);
  }
  ~SyntheticFile() { std::filesystem::remove(path_); }
  SyntheticFile(const SyntheticFile&) = delete;
  SyntheticFile& operator=(const SyntheticFile&) = delete;

  [[nodiscard]] const std::string& Path() const { return path_; }
  [[nodiscard]] const Synthetic& Data() const { return synthetic_; }

private:
  Synthetic synthetic_;
  std::string path_ = "synthetic.geqdsk";
};

// Points inside the plasma (0 <= psi <= 0.1): (R, phi, Z) rows.
const std::vector<Real> Inside = {1.55, 0.0, 0.1, 1.45, 1.3, -0.2,
                                  1.5,  2.0, 0.0, 1.62, 5.0, 0.15};

} // namespace

TEST_CASE("geqdsk: the quantities of an analytic equilibrium")
{
  const Real sign = GENERATE(1.0, -1.0);
  const SyntheticFile file(sign);
  const Synthetic& eq = file.Data();
  const auto source = fusionio::OpenGeqdsk(file.Path());
  constexpr Real Tol = 1e-11;
  const Real simag = sign > 0 ? 0 : 0.1, sibry = sign > 0 ? 0.1 : 0;

  const auto psi = source->CreateFunction(Quantity::PoloidalFlux);
  const auto norm = source->CreateFunction(Quantity::PoloidalFluxNorm);
  const auto pressure = source->CreateFunction(Quantity::TotalPressure);
  const auto b = source->CreateFunction(Quantity::MagneticField);
  const auto psi_v = EvaluateAt(psi, Inside);
  const auto norm_v = EvaluateAt(norm, Inside);
  const auto p_v = EvaluateAt(pressure, Inside);
  const auto b_v = EvaluateAt(b, Inside);
  const auto grad_v = fusionio::test::EvaluateGradientAt(psi, Inside);
  const auto norm_grad_v = fusionio::test::EvaluateGradientAt(norm, Inside);

  for (size_t i = 0; i < Inside.size() / 3; ++i) {
    const Real r = Inside[3 * i], z = Inside[3 * i + 2];
    const Real flux = eq.Psi(r, z);
    CAPTURE(i, r, z);
    REQUIRE_THAT(psi_v(i, 0), WithinAbs(2 * std::numbers::pi * flux, Tol));
    REQUIRE_THAT(norm_v(i, 0),
                 WithinAbs((flux - simag) / (sibry - simag), Tol));
    REQUIRE_THAT(p_v(i, 0), WithinRel(Synthetic::P(flux), Tol));
    // B = F grad(phi) + grad(phi) x grad(psi)
    REQUIRE_THAT(b_v(i, 0), WithinAbs(eq.PsiZ(z) / r, Tol));
    REQUIRE_THAT(b_v(i, 1), WithinRel(Synthetic::F(flux) / r, Tol));
    REQUIRE_THAT(b_v(i, 2), WithinAbs(-eq.PsiR(r) / r, Tol));
    // gradients: (d/dR, (1/R) d/dphi, d/dZ)
    REQUIRE_THAT(grad_v(i, 0),
                 WithinAbs(2 * std::numbers::pi * eq.PsiR(r), Tol));
    REQUIRE_THAT(grad_v(i, 1), WithinAbs(0, Tol));
    REQUIRE_THAT(grad_v(i, 2),
                 WithinAbs(2 * std::numbers::pi * eq.PsiZ(z), Tol));
    REQUIRE_THAT(norm_grad_v(i, 0),
                 WithinAbs(eq.PsiR(r) / (sibry - simag), Tol));
    REQUIRE_THAT(norm_grad_v(i, 2),
                 WithinAbs(eq.PsiZ(z) / (sibry - simag), Tol));
  }

  SECTION("the profiles")
  {
    // 1D functions of psi (Wb/rad, the file's flux)
    const auto f =
      source->CreateProfile(fusionio::GeqdskProfile::PoloidalCurrent);
    const auto p = source->CreateProfile(fusionio::GeqdskProfile::Pressure);
    const auto q = source->CreateProfile(fusionio::GeqdskProfile::SafetyFactor);
    const std::vector<Real> fluxes = {0.0, 0.013, 0.05, 0.0999, 0.1};
    const auto f_v = EvaluateAt(f, fluxes);
    const auto p_v = EvaluateAt(p, fluxes);
    const auto q_v = EvaluateAt(q, fluxes);
    for (size_t i = 0; i < fluxes.size(); ++i) {
      CAPTURE(fluxes[i]);
      REQUIRE_THAT(f_v(i, 0), WithinRel(Synthetic::F(fluxes[i]), Tol));
      REQUIRE_THAT(p_v(i, 0), WithinRel(Synthetic::P(fluxes[i]), Tol));
      REQUIRE_THAT(q_v(i, 0), WithinRel(Synthetic::Q(fluxes[i]), Tol));
    }
  }

  SECTION("profiles are constant outside the plasma")
  {
    // psi(1.95, 0.8) is beyond the boundary's 0.1 on a flux that increases
    // outward, and below the axis for the reversed file
    const std::vector<Real> outside = {1.95, 0.0, 0.8};
    const Real flux = eq.Psi(1.95, 0.8);
    const Real end = sibry;
    REQUIRE(((flux - end) * sign > 0));
    REQUIRE_THAT(EvaluateAt(pressure, outside)(0, 0),
                 WithinRel(Synthetic::P(end), Tol));
    REQUIRE_THAT(EvaluateAt(b, outside)(0, 1),
                 WithinRel(Synthetic::F(end) / 1.95, Tol));
  }
}

TEST_CASE("geqdsk: the source describes the file")
{
  const SyntheticFile file(1.0);
  const auto source = fusionio::OpenGeqdsk(file.Path());
  using fusionio::Series;

  REQUIRE(source->Times().empty());
  REQUIRE_THAT(source->ToroidalPeriod(),
               WithinRel(2 * std::numbers::pi, 1e-14));
  REQUIRE(*source->EvaluateSeries(Series::MagneticAxisR, 0) == 1.5);
  REQUIRE(*source->EvaluateSeries(Series::MagneticAxisZ, 0) == 0.0);
  REQUIRE(*source->EvaluateSeries(Series::MagneticAxisPsi, 0) == 0.0);
  REQUIRE(*source->EvaluateSeries(Series::LcfsPsi, 0) == 0.1);
  const auto box = source->BoundingBox();
  REQUIRE(box.has_value());
  REQUIRE(box->min == fusionio::Point{1.0, 0.0, -1.0});
  REQUIRE_THAT(box->max[1], WithinRel(2 * std::numbers::pi, 1e-14));
  REQUIRE(box->max[0] == 2.0);
  REQUIRE(box->max[2] == 1.0);
  REQUIRE(source->Quantities().size() == 4);

  // the curves are plain data
  REQUIRE(source->Boundary().size() == 5);
  REQUIRE(source->Boundary()[1] == std::array<Real, 2>{1.5, 0.5});
  REQUIRE(source->Limiter().size() == 4);
  REQUIRE(source->Limiter()[2] == std::array<Real, 2>{1.9, 0.9});
}

TEST_CASE("geqdsk: points outside the grid, unknown quantities and options")
{
  const SyntheticFile file(1.0);
  const auto source = fusionio::OpenGeqdsk(file.Path());
  const auto psi = source->CreateFunction(Quantity::PoloidalFlux);

  // outside the (R, Z) grid: an error unless the points are filled
  REQUIRE_THROWS(EvaluateAt(psi, {5.0, 0.0, 0.0}));
  constexpr Real Fill = -7.0;
  const auto& space = psi.GetSpace();
  const auto bound = fusionio::test::Bind(
    *source->GetGeometry(), {1.55, 0.0, 0.1, 5.0, 0.0, 0.0, 1.45, 0.0, -0.2},
    pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, Fill});
  const auto evaluator = space.CreatePointEvaluator<Real>(bound);
  const auto values = fusionio::test::EvaluateWith(*evaluator, psi);
  REQUIRE(values(1, 0) == Fill);
  REQUIRE_THAT(
    values(0, 0),
    WithinAbs(2 * std::numbers::pi * Synthetic{1.0}.Psi(1.55, 0.1), 1e-11));

  REQUIRE_THROWS_WITH(source->CreateFunction(Quantity::Viscosity),
                      ContainsSubstring("\"viscosity\" is not a quantity") &&
                        ContainsSubstring("magnetic field"));
  REQUIRE_THROWS_WITH(
    source->CreateFunction(Quantity::MagneticField, {.timeslice = 0}),
    ContainsSubstring("no options"));
  // the gradient of the magnetic field and of the pressure is not provided
  const auto b = source->CreateFunction(Quantity::MagneticField);
  REQUIRE_FALSE(b.GetSpace().Supports(pcms::DifferentialOperator::Gradient));
  REQUIRE(psi.GetSpace().Supports(pcms::DifferentialOperator::Gradient));
}

TEST_CASE("geqdsk: Open reads the file natively on request")
{
  const SyntheticFile file(1.0);
  const auto source = fusionio::Open(fusionio::Format::GEQDSK, file.Path(),
                                     fusionio::Evaluation::Native);
  REQUIRE(dynamic_cast<fusionio::GeqdskSource*>(source.get()) != nullptr);
}

TEST_CASE("geqdsk: malformed files are reported")
{
  REQUIRE_THROWS_WITH(fusionio::OpenGeqdsk("missing.geqdsk"),
                      ContainsSubstring("cannot open"));

  const std::string path = "malformed.geqdsk";
  const auto write = [&](const std::string& text) {
    std::ofstream(path) << text;
  };
  write("no sizes here\n");
  REQUIRE_THROWS_WITH(fusionio::OpenGeqdsk(path),
                      ContainsSubstring("grid's sizes"));
  write("EFIT 01/01/2026 #0 0ms 3 2 2\n");
  REQUIRE_THROWS_WITH(fusionio::OpenGeqdsk(path),
                      ContainsSubstring("smaller than 4 x 4"));
  write("EFIT 01/01/2026 #0 0ms 3 8 8\n1 2 3\n");
  REQUIRE_THROWS_WITH(fusionio::OpenGeqdsk(path),
                      ContainsSubstring("ends early"));
  std::filesystem::remove(path);
}
