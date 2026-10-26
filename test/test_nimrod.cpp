#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <pcms/field/function_space/nimrod.hpp>
#include <pcms/field/function_space/nimeval.hpp>
#include "field_test_utils.h"
#include <unistd.h>
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
using pcms::DeviceMemorySpace;
using pcms::HostMemorySpace;
using pcms::NimrodOptions;
using pcms::Real;

namespace
{

namespace fs = std::filesystem;

// A NIMROD case directory (nimrod.in and dump) that is not distributed with
// the test data; set PCMS_TEST_NIMROD_DIR to a copy of the cylrwtm case.
const fs::path CaseDir = PCMS_TEST_NIMROD_DIR;
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

// Values of `field` at `points`, (R, theta, Z) row-major, copied to the host.
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
  auto source = pcms::OpenNimrodSource(copy.DumpPath(), "bvn", copy.Dir());
  auto scalars = pcms::NimrodFunctionSpace::Create(source, 1);
  auto vectors = pcms::NimrodFunctionSpace::Create(source, 3);

  struct Case
  {
    const char* quantity;
    NimrodOptions::Part part;
    int eq;
    std::shared_ptr<pcms::NimrodFunctionSpace> space;
  };
  for (const auto& [quantity, part, eq, space] :
       {Case{"b", NimrodOptions::Part::Total, 2, vectors},
        Case{"v", NimrodOptions::Part::Perturbed, 0, vectors},
        Case{"n", NimrodOptions::Part::Equilibrium, 1, scalars}}) {
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
                     WithinRel(expected[c], pcms::test::ExactTol) ||
                       WithinAbs(expected[c], pcms::test::ExactTol));
      }
    }
  }
}

TEST_CASE("NIMROD parts select equilibrium and perturbation")
{
  const CaseCopy copy("parts");
  auto source = pcms::OpenNimrodSource(copy.DumpPath(), "b", copy.Dir());
  auto space = pcms::NimrodFunctionSpace::Create(source, 3);
  // Two points in the same poloidal plane position, half a toroidal turn
  // apart.
  const std::vector<Real> points = {
    10.6, 0.4, 0.3, 10.6, 0.4 + std::numbers::pi, 0.3};

  const auto equilibrium = EvaluateAt(
    *space,
    space->CreateFunction("b", "b", {.part = NimrodOptions::Part::Equilibrium}),
    points);
  const auto perturbed = EvaluateAt(
    *space,
    space->CreateFunction("b", "b", {.part = NimrodOptions::Part::Perturbed}),
    points);
  const auto total =
    EvaluateAt(*space, space->CreateFunction("b", "b"), points);
  for (int c = 0; c < 3; ++c) {
    CAPTURE(c);
    // The equilibrium is axisymmetric and the case's n = 1 perturbation
    // changes sign over half a turn.
    REQUIRE_THAT(equilibrium(1, c),
                 WithinAbs(equilibrium(0, c), pcms::test::EvalTol));
    REQUIRE_THAT(perturbed(1, c),
                 WithinAbs(-perturbed(0, c), pcms::test::EvalTol));
    REQUIRE_THAT(total(0, c), WithinAbs(equilibrium(0, c) + perturbed(0, c),
                                        pcms::test::EvalTol));
  }
  REQUIRE(perturbed(0, 0) != 0.0);
}

TEST_CASE("NIMROD evaluation errors")
{
  const CaseCopy copy("errors");
  auto source = pcms::OpenNimrodSource(copy.DumpPath(), "bn", copy.Dir());
  auto scalars = pcms::NimrodFunctionSpace::Create(source, 1);
  auto vectors = pcms::NimrodFunctionSpace::Create(source, 3);
  auto n = scalars->CreateFunction("n", "n");
  REQUIRE(pcms::AvailableQuantities(*source) ==
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
  const auto coords = pcms::test::CreateDeviceCoordinateView(
    {10.6, 0.0, 0.3, 10.2, 0.0, 0.1}, scalars->GetCoordinateSystem(), 3);
  const auto evaluator =
    scalars->CreatePointEvaluator<Real>(scalars->Bind(coords.coordinate_view));
  Kokkos::View<Real**, DeviceMemorySpace> short_out("short_out", 1, 1);
  REQUIRE_THROWS_WITH(evaluator->Evaluate(n, pcms::test::TagLike(n, short_out)),
                      ContainsSubstring("but expected [2][1]"));

  // nimeval holds one dump per process.
  REQUIRE_THROWS_WITH(pcms::OpenNimrodSource(copy.DumpPath(), "bn", copy.Dir()),
                      ContainsSubstring("another NIMROD source is open"));
}

TEST_CASE("NIMROD sources are checked before opening")
{
  const CaseCopy copy("open");
  REQUIRE_THROWS_WITH(pcms::OpenNimrodSource(copy.DumpPath(), "be", copy.Dir()),
                      ContainsSubstring("unsupported field id 'e'"));
  REQUIRE_THROWS_WITH(pcms::OpenNimrodSource(copy.DumpPath(), "", copy.Dir()),
                      ContainsSubstring("no field ids"));
  REQUIRE_THROWS_WITH(pcms::OpenNimrodSource("missing.h5", "b", copy.Dir()),
                      ContainsSubstring("cannot find dump"));
  REQUIRE_THROWS_WITH(
    pcms::OpenNimrodSource(copy.DumpPath(), "b", copy.Dir() / "missing"),
    ContainsSubstring("cannot read"));

  // A slab case.
  std::stringstream text;
  text << std::ifstream(copy.Dir() / "nimrod.in").rdbuf();
  std::string input = text.str();
  input.replace(input.find("geom='tor'"), 10, "geom='lin'");
  std::ofstream(copy.Dir() / "nimrod.in") << input;
  REQUIRE_THROWS_WITH(pcms::OpenNimrodSource(copy.DumpPath(), "b", copy.Dir()),
                      ContainsSubstring("only toroidal cases"));
}
