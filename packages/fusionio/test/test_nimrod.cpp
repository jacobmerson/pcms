#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <fusionio/fusionio.hpp>
#include "nimeval.hpp"
#include "test_utils.hpp"
#include <unistd.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using fusionio::Quantity;
using fusionio::test::EvaluateAt;
using fusionio::test::EvaluateGradientAt;
using pcms::DeviceMemorySpace;
using pcms::HostMemorySpace;
using pcms::Real;

namespace
{

namespace fs = std::filesystem;

// A NIMROD case directory (nimrod.in and dump) that is not distributed with
// the test data; set FUSIONIO_TEST_NIMROD_DIR to a copy of the cylrwtm case.
const fs::path CaseDir = FUSIONIO_TEST_NIMROD_DIR;
const std::string Dump = "dump_gs.00000.h5";

// A writable copy of the case, which NIMROD needs for its scratch files.
class CaseCopy
{
public:
  explicit CaseCopy(const std::string& name)
    : dir_(fs::temp_directory_path() /
           ("pcms_nimrod_" + name + "_" + std::to_string(getpid())))
  {
    fs::create_directories(dir_);
    fs::copy_file(CaseDir / "nimrod.in", dir_ / "nimrod.in",
                  fs::copy_options::overwrite_existing);
    fs::copy_file(CaseDir / Dump, dir_ / Dump,
                  fs::copy_options::overwrite_existing);
  }
  ~CaseCopy()
  {
    std::error_code ignored;
    fs::remove_all(dir_, ignored);
  }
  CaseCopy(const CaseCopy&) = delete;
  CaseCopy& operator=(const CaseCopy&) = delete;

  [[nodiscard]] const fs::path& Dir() const noexcept { return dir_; }
  [[nodiscard]] std::string DumpPath() const { return (dir_ / Dump).string(); }

private:
  fs::path dir_;
};

// nimeval's own value of `quantity` at its (R, Z, phi) point `rzp`.
std::vector<double> EvaluateDirectly(const std::string& quantity,
                                     std::array<double, 3> rzp, int nqty,
                                     int eq)
{
  std::vector<double> values(nqty);
  std::array<double, 2> guess{-1.0, -1.0};
  int ifail = 0;
  const int dmode = 0;
  c_eval_field_dm(rzp.data(), quantity.c_str(), values.data(), &nqty,
                  guess.data(), &ifail, &dmode, &eq);
  REQUIRE(ifail == 0);
  return values;
}

// Points inside the cylrwtm domain, which extends past 1.2 from (R, Z) = (10,
// 0).
const std::vector<Real> Points = {10.6, -0.4, 0.3, 9.5,  -2.0, -0.5,
                                  10.2, 0.7,  0.1, 10.9, 3.0,  -0.6};

} // namespace

TEST_CASE("NIMROD values match direct nimeval evaluation")
{
  const CaseCopy copy("direct");
  const auto source =
    fusionio::OpenNimrod(copy.DumpPath(), "bvn", copy.Dir().string());

  struct Case
  {
    Quantity quantity;
    const char* name;
    fusionio::Part part;
    int eq;
  };
  for (const auto& [quantity, name, part, eq] :
       {Case{Quantity::MagneticField, "b", fusionio::Part::Total, 2},
        Case{Quantity::FluidVelocity, "v", fusionio::Part::Perturbed, 0},
        Case{Quantity::Density, "n", fusionio::Part::Equilibrium, 1}}) {
    CAPTURE(name);
    const auto function = source->CreateFunction(quantity, {.part = part});
    const auto values = EvaluateAt(function, Points);
    const int nqty = function.GetLayout().GetNumComponents();
    for (size_t i = 0; i < values.extent(0); ++i) {
      // NIMROD's points are (R, Z, phi) with phi = -theta and its vectors
      // (v_R, v_Z, v_phi), so v_theta = -v_phi.
      auto expected = EvaluateDirectly(
        name, {Points[3 * i], Points[3 * i + 2], -Points[3 * i + 1]}, nqty, eq);
      if (nqty == 3) {
        expected = {expected[0], -expected[2], expected[1]};
      }
      for (int c = 0; c < nqty; ++c) {
        CAPTURE(i, c);
        // The absolute floor covers components near zero.
        REQUIRE_THAT(values(i, c),
                     WithinRel(expected[c], fusionio::test::ExactTol) ||
                       WithinAbs(expected[c], fusionio::test::ExactTol));
      }
    }
  }
}

TEST_CASE("NIMROD parts select equilibrium and perturbation")
{
  const CaseCopy copy("parts");
  const auto source =
    fusionio::OpenNimrod(copy.DumpPath(), "b", copy.Dir().string());
  // Two points in the same poloidal plane position, half a toroidal turn
  // apart.
  const std::vector<Real> points = {
    10.6, 0.4, 0.3, 10.6, 0.4 + std::numbers::pi, 0.3};

  const auto equilibrium =
    EvaluateAt(source->CreateFunction(Quantity::MagneticField,
                                      {.part = fusionio::Part::Equilibrium}),
               points);
  const auto perturbed =
    EvaluateAt(source->CreateFunction(Quantity::MagneticField,
                                      {.part = fusionio::Part::Perturbed}),
               points);
  const auto total =
    EvaluateAt(source->CreateFunction(Quantity::MagneticField), points);
  for (int c = 0; c < 3; ++c) {
    CAPTURE(c);
    // The equilibrium is axisymmetric and the case's n = 1 perturbation
    // changes sign over half a turn.
    REQUIRE_THAT(equilibrium(1, c),
                 WithinAbs(equilibrium(0, c), fusionio::test::EvalTol));
    REQUIRE_THAT(perturbed(1, c),
                 WithinAbs(-perturbed(0, c), fusionio::test::EvalTol));
    REQUIRE_THAT(total(0, c), WithinAbs(equilibrium(0, c) + perturbed(0, c),
                                        fusionio::test::EvalTol));
  }
  REQUIRE(perturbed(0, 0) != 0.0);
}

TEST_CASE("NIMROD gradients match direct nimeval evaluation")
{
  const CaseCopy copy("gradient");
  const auto source =
    fusionio::OpenNimrod(copy.DumpPath(), "bn", copy.Dir().string());
  // The cylrwtm case has no scalar with a physical gradient (its density is
  // uniform, its pressure and temperatures zero), so this checks only that
  // NIMROD's (d/dR, d/dZ, (1/R) d/dphi) is reordered to (R, theta, Z) with
  // phi = -theta, against nimeval's own dmode 1 output. The density's
  // gradient is round-off that depends on the search's starting guess, so
  // each point is bound alone and, like the direct call, searched from
  // scratch.
  const auto n = source->CreateFunction(Quantity::Density);
  REQUIRE(n.GetLayout().GetNumComponents() == 1);
  const std::vector<Real> points = {10.6, 0.4, 0.3, 10.2, 2.0, -0.1};
  for (size_t i = 0; i < points.size() / 3; ++i) {
    const auto grad = EvaluateGradientAt(
      n, {points[3 * i], points[3 * i + 1], points[3 * i + 2]});
    std::array<double, 3> rzp{points[3 * i], points[3 * i + 2],
                              -points[3 * i + 1]};
    std::array<double, 4> raw{};
    std::array<double, 2> guess{-1.0, -1.0};
    int nqty = 1, ifail = 0, eq = 2;
    const int dmode = 1;
    c_eval_field_dm(rzp.data(), "n", raw.data(), &nqty, guess.data(), &ifail,
                    &dmode, &eq);
    REQUIRE(ifail == 0);
    const double expected[3] = {raw[1], -raw[3], raw[2]};
    for (int c = 0; c < 3; ++c) {
      CAPTURE(i, c);
      REQUIRE_THAT(grad(0, c),
                   WithinRel(expected[c], fusionio::test::ExactTol) ||
                     WithinAbs(expected[c], fusionio::test::ExactTol));
    }
  }

  const auto b = source->CreateFunction(Quantity::MagneticField);
  const auto bound = fusionio::test::Bind(*source->GetGeometry(), points);
  REQUIRE(b.GetSpace().GetGeometry() == source->GetGeometry());
  REQUIRE_THROWS_WITH(b.GetSpace().CreatePointEvaluator<Real>(
                        bound, pcms::DifferentialOperator::Gradient),
                      ContainsSubstring("not supported yet"));
  REQUIRE_THROWS_WITH(n.GetSpace().CreatePointEvaluator<Real>(
                        bound, pcms::DifferentialOperator::Hessian),
                      ContainsSubstring("Hessian is not supported"));
}

TEST_CASE("NIMROD fills points outside the domain on request")
{
  const CaseCopy copy("fill");
  const auto source =
    fusionio::OpenNimrod(copy.DumpPath(), "bn", copy.Dir().string());
  const auto b = source->CreateFunction(Quantity::MagneticField);
  const auto n = source->CreateFunction(Quantity::Density);
  // The second point is outside the domain.
  const std::vector<Real> points = {10.6, -0.4, 0.3, 20.0, 0.0,
                                    0.0,  10.2, 0.7, 0.1};
  constexpr Real Fill = -7.0;
  const auto bound = fusionio::test::Bind(*source->GetGeometry(), points,
                                          {pcms::OutOfBoundsMode::FILL, Fill});
  REQUIRE(bound->MissingPoints().extent(0) == 1);
  for (const auto* f : {&b, &n}) {
    const auto values = fusionio::test::EvaluateWith(
      *f->GetSpace().CreatePointEvaluator<Real>(bound), *f);
    const auto expected = EvaluateAt(*f, {10.6, -0.4, 0.3, 10.2, 0.7, 0.1});
    for (size_t c = 0; c < values.extent(1); ++c) {
      CAPTURE(f->GetName(), c);
      REQUIRE(values(1, c) == Fill);
      REQUIRE_THAT(values(0, c),
                   WithinRel(expected(0, c), fusionio::test::ExactTol) ||
                     WithinAbs(expected(0, c), fusionio::test::ExactTol));
      REQUIRE_THAT(values(2, c),
                   WithinRel(expected(1, c), fusionio::test::ExactTol) ||
                     WithinAbs(expected(1, c), fusionio::test::ExactTol));
    }
  }
  REQUIRE_THROWS_WITH(
    fusionio::test::Bind(*source->GetGeometry(), points,
                         {pcms::OutOfBoundsMode::NEAREST_BOUNDARY}),
    ContainsSubstring("NEAREST_BOUNDARY"));
}

TEST_CASE("NIMROD evaluation errors")
{
  const CaseCopy copy("errors");
  const auto source =
    fusionio::OpenNimrod(copy.DumpPath(), "bn", copy.Dir().string());
  const auto n = source->CreateFunction(Quantity::Density);

  REQUIRE_THROWS_WITH(EvaluateAt(n, {20.0, 0.0, 0.0}),
                      ContainsSubstring("outside the dump's domain"));
  // The evaluator's search guess recovers from a failed point.
  REQUIRE_NOTHROW(EvaluateAt(n, {10.6, 0.0, 0.3}));
  REQUIRE_THROWS_WITH(source->CreateFunction(Quantity::TotalPressure),
                      ContainsSubstring("NIMROD's \"p\""));
  REQUIRE_THROWS_WITH(n.GetDOFHolderData(), ContainsSubstring("no DOFAccess"));

  // NIMROD writes one row per query point.
  const auto evaluator =
    n.GetSpace().CreatePointEvaluator<Real>(fusionio::test::Bind(
      *source->GetGeometry(), {10.6, 0.0, 0.3, 10.2, 0.0, 0.1}));
  Kokkos::View<Real**, DeviceMemorySpace> short_out("short_out", 1, 1);
  REQUIRE_THROWS_WITH(
    evaluator->Evaluate(n, fusionio::test::TagLike(n, short_out)),
    ContainsSubstring("but must be [2][1]"));

  // nimeval holds one dump per process.
  REQUIRE_THROWS_WITH(
    fusionio::OpenNimrod(copy.DumpPath(), "bn", copy.Dir().string()),
    ContainsSubstring("another NIMROD source is open"));
}

TEST_CASE("NIMROD sources are checked before opening")
{
  const CaseCopy copy("open");
  REQUIRE_THROWS_WITH(fusionio::OpenNimrod(copy.DumpPath(), "be", copy.Dir()),
                      ContainsSubstring("unsupported field id 'e'"));
  REQUIRE_THROWS_WITH(fusionio::OpenNimrod(copy.DumpPath(), "", copy.Dir()),
                      ContainsSubstring("no field ids"));
  REQUIRE_THROWS_WITH(fusionio::OpenNimrod("missing.h5", "b", copy.Dir()),
                      ContainsSubstring("cannot find dump"));
  REQUIRE_THROWS_WITH(
    fusionio::OpenNimrod(copy.DumpPath(), "b", copy.Dir() / "missing"),
    ContainsSubstring("cannot read"));

  // A slab case.
  std::stringstream text;
  text << std::ifstream(copy.Dir() / "nimrod.in").rdbuf();
  std::string input = text.str();
  input.replace(input.find("geom='tor'"), 10, "geom='lin'");
  std::ofstream(copy.Dir() / "nimrod.in") << input;
  REQUIRE_THROWS_WITH(fusionio::OpenNimrod(copy.DumpPath(), "b", copy.Dir()),
                      ContainsSubstring("only toroidal cases"));
}

TEST_CASE("fusionio::OpenNimrod gives Fusion-IO's quantities")
{
  const CaseCopy copy("fusionio");
  const auto source = fusionio::OpenNimrod(copy.DumpPath(), "bvn", copy.Dir());

  const auto quantities = source->Quantities();
  const auto components = [&](fusionio::Quantity quantity) {
    const auto found = std::ranges::find(quantities, quantity,
                                         &fusionio::QuantityInfo::quantity);
    return found == quantities.end() ? 0 : found->num_components;
  };
  REQUIRE(components(fusionio::Quantity::MagneticField) == 3);
  REQUIRE(components(fusionio::Quantity::FluidVelocity) == 3);
  REQUIRE(components(fusionio::Quantity::Density) >= 1);
  REQUIRE(components(fusionio::Quantity::Temperature) == 0);
  REQUIRE(source->Times().empty());

  SECTION("functions share the source's geometry and spaces")
  {
    const auto b = source->CreateFunction(
      fusionio::Quantity::MagneticField, {.part = fusionio::Part::Equilibrium});
    const auto v = source->CreateFunction(fusionio::Quantity::FluidVelocity);
    REQUIRE(b.GetName() == "magnetic field");
    REQUIRE(b.GetData().GetValueBasis().units.has_value());
    REQUIRE(b.GetMetadata().duality == pcms::Duality::Primal);
    REQUIRE(b.GetSpace().GetGeometry() == source->GetGeometry());
    REQUIRE(&b.GetSpace() == &v.GetSpace());
    REQUIRE(b.GetSpace().Supports(pcms::DifferentialOperator::Gradient));
  }

  SECTION("unsupported options and quantities throw")
  {
    REQUIRE_THROWS_WITH(source->CreateFunction(
                          fusionio::Quantity::MagneticField, {.timeslice = 0}),
                        ContainsSubstring("only the part option"));
    REQUIRE_THROWS_WITH(source->CreateFunction(fusionio::Quantity::Temperature),
                        ContainsSubstring("does not provide"));
    REQUIRE_THROWS_WITH(source->CreateFunction(fusionio::Quantity::Viscosity),
                        ContainsSubstring("do not provide"));
  }
}
