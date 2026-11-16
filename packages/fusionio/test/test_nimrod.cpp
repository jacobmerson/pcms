#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include "nimrod_space.hpp"
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
using fusionio::detail::NimrodOptions;
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
  auto source =
    fusionio::detail::OpenNimrodSource(copy.DumpPath(), "bvn", copy.Dir());
  auto scalars = fusionio::detail::NimrodFunctionSpace::Create(source, 1);
  auto vectors = fusionio::detail::NimrodFunctionSpace::Create(source, 3);

  struct Case
  {
    const char* quantity;
    fusionio::Part part;
    int eq;
    std::shared_ptr<fusionio::detail::NimrodFunctionSpace> space;
  };
  for (const auto& [quantity, part, eq, space] :
       {Case{"b", fusionio::Part::Total, 2, vectors},
        Case{"v", fusionio::Part::Perturbed, 0, vectors},
        Case{"n", fusionio::Part::Equilibrium, 1, scalars}}) {
    CAPTURE(quantity);
    const auto values = EvaluateAt(
      *space, space->CreateFunction("f", quantity, {.part = part}), Points);
    const int nqty = space->GetLayout()->GetNumComponents();
    for (size_t i = 0; i < values.extent(0); ++i) {
      // NIMROD's points are (R, Z, phi) with phi = -theta and its vectors
      // (v_R, v_Z, v_phi), so v_theta = -v_phi.
      auto expected = EvaluateDirectly(
        quantity, {Points[3 * i], Points[3 * i + 2], -Points[3 * i + 1]}, nqty,
        eq);
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
  auto source =
    fusionio::detail::OpenNimrodSource(copy.DumpPath(), "b", copy.Dir());
  auto space = fusionio::detail::NimrodFunctionSpace::Create(source, 3);
  // Two points in the same poloidal plane position, half a toroidal turn
  // apart.
  const std::vector<Real> points = {
    10.6, 0.4, 0.3, 10.6, 0.4 + std::numbers::pi, 0.3};

  const auto equilibrium = EvaluateAt(
    *space,
    space->CreateFunction("b", "b", {.part = fusionio::Part::Equilibrium}),
    points);
  const auto perturbed = EvaluateAt(
    *space,
    space->CreateFunction("b", "b", {.part = fusionio::Part::Perturbed}),
    points);
  const auto total =
    EvaluateAt(*space, space->CreateFunction("b", "b"), points);
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
  auto source =
    fusionio::detail::OpenNimrodSource(copy.DumpPath(), "bn", copy.Dir());
  auto scalars = fusionio::detail::NimrodFunctionSpace::Create(source, 1);
  auto vectors = fusionio::detail::NimrodFunctionSpace::Create(source, 3);
  // The cylrwtm case has no scalar with a physical gradient (its density is
  // uniform, its pressure and temperatures zero), so this checks only that
  // NIMROD's (d/dR, d/dZ, (1/R) d/dphi) is reordered to (R, theta, Z) with
  // phi = -theta, against nimeval's own dmode 1 output. The density's
  // gradient is round-off that depends on the search's starting guess, so
  // each point is bound alone and, like the direct call, searched from
  // scratch.
  auto n = scalars->CreateFunction("n", "n");
  const std::vector<Real> points = {10.6, 0.4, 0.3, 10.2, 2.0, -0.1};
  for (size_t i = 0; i < points.size() / 3; ++i) {
    const auto grad = EvaluateGradientAt(
      *scalars, n, {points[3 * i], points[3 * i + 1], points[3 * i + 2]});
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

  const auto coords = fusionio::test::CreateDeviceCoordinateView(
    points, vectors->GetCoordinateSystem(), 3);
  REQUIRE_THROWS_WITH(vectors->CreatePointEvaluator<Real>(
                        vectors->Bind(coords.coordinate_view), 1),
                      ContainsSubstring("not supported yet"));
  REQUIRE_THROWS_WITH(scalars->CreatePointEvaluator<Real>(
                        scalars->Bind(coords.coordinate_view), 2),
                      ContainsSubstring("derivative order 2"));
}

TEST_CASE("NIMROD evaluation errors")
{
  const CaseCopy copy("errors");
  auto source =
    fusionio::detail::OpenNimrodSource(copy.DumpPath(), "bn", copy.Dir());
  auto scalars = fusionio::detail::NimrodFunctionSpace::Create(source, 1);
  auto vectors = fusionio::detail::NimrodFunctionSpace::Create(source, 3);
  auto n = scalars->CreateFunction("n", "n");
  REQUIRE(fusionio::detail::AvailableQuantities(*source) ==
          std::vector<std::string>{"b", "n"});

  REQUIRE_THROWS_WITH(EvaluateAt(*scalars, n, {20.0, 0.0, 0.0}),
                      ContainsSubstring("outside the dump's domain"));
  // The evaluator's search guess recovers from a failed point.
  REQUIRE_NOTHROW(EvaluateAt(*scalars, n, {10.6, 0.0, 0.3}));
  REQUIRE_THROWS_WITH(scalars->CreateFunction("B", "b"),
                      ContainsSubstring("the space holds 1"));
  REQUIRE_THROWS_WITH(vectors->CreateFunction("n", "n"),
                      ContainsSubstring("the space holds 3"));
  REQUIRE_THROWS_WITH(scalars->CreateFunction("p", "p"),
                      ContainsSubstring("\"p\" is not a quantity") &&
                        ContainsSubstring("it has: b, n"));
  REQUIRE_THROWS_WITH(n.GetDOFHolderData(), ContainsSubstring("no DOFAccess"));

  // NIMROD writes one row per query point.
  const auto coords = fusionio::test::CreateDeviceCoordinateView(
    {10.6, 0.0, 0.3, 10.2, 0.0, 0.1}, scalars->GetCoordinateSystem(), 3);
  const auto evaluator =
    scalars->CreatePointEvaluator<Real>(scalars->Bind(coords.coordinate_view));
  Kokkos::View<Real**, DeviceMemorySpace> short_out("short_out", 1, 1);
  REQUIRE_THROWS_WITH(
    evaluator->Evaluate(n, fusionio::test::TagLike(n, short_out)),
    ContainsSubstring("but must be [2][1]"));

  // nimeval holds one dump per process.
  REQUIRE_THROWS_WITH(
    fusionio::detail::OpenNimrodSource(copy.DumpPath(), "bn", copy.Dir()),
    ContainsSubstring("another NIMROD source is open"));
}

TEST_CASE("NIMROD sources are checked before opening")
{
  const CaseCopy copy("open");
  REQUIRE_THROWS_WITH(
    fusionio::detail::OpenNimrodSource(copy.DumpPath(), "be", copy.Dir()),
    ContainsSubstring("unsupported field id 'e'"));
  REQUIRE_THROWS_WITH(
    fusionio::detail::OpenNimrodSource(copy.DumpPath(), "", copy.Dir()),
    ContainsSubstring("no field ids"));
  REQUIRE_THROWS_WITH(
    fusionio::detail::OpenNimrodSource("missing.h5", "b", copy.Dir()),
    ContainsSubstring("cannot find dump"));
  REQUIRE_THROWS_WITH(fusionio::detail::OpenNimrodSource(
                        copy.DumpPath(), "b", copy.Dir() / "missing"),
                      ContainsSubstring("cannot read"));

  // A slab case.
  std::stringstream text;
  text << std::ifstream(copy.Dir() / "nimrod.in").rdbuf();
  std::string input = text.str();
  input.replace(input.find("geom='tor'"), 10, "geom='lin'");
  std::ofstream(copy.Dir() / "nimrod.in") << input;
  REQUIRE_THROWS_WITH(
    fusionio::detail::OpenNimrodSource(copy.DumpPath(), "b", copy.Dir()),
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

  SECTION("functions match the NIMROD function space")
  {
    const auto b = source->CreateFunction(
      fusionio::Quantity::MagneticField, {.part = fusionio::Part::Equilibrium});
    REQUIRE(b.GetName() == "magnetic field");
    const auto& space =
      dynamic_cast<const fusionio::detail::NimrodFunctionSpace&>(b.GetSpace());
    const auto expected =
      space.CreateFunction("b", "b", {.part = fusionio::Part::Equilibrium});
    const auto actual_values = fusionio::test::EvaluateAt(b, Points);
    const auto expected_values = fusionio::test::EvaluateAt(expected, Points);
    for (size_t i = 0; i < Points.size() / 3; ++i) {
      for (size_t c = 0; c < 3; ++c) {
        REQUIRE(actual_values(i, c) == expected_values(i, c));
      }
    }
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
