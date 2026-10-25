// NIMROD field evaluation validation test.
//
// Requires a NIMROD build (PCMS_ENABLE_NIMROD=ON, linking `nimeval`). The test
// is run from the prepared case directory (set as the CTest WORKING_DIRECTORY)
// so NIMROD finds the dump and its nimrod.in.
//
// Validation against nimpy: nimpy's EvalNimrod wraps the same c_eval_field_dm
// symbol this backend calls, so agreement is expected to round-off. Reference
// values from nimpy EvalNimrod.eval_field(field, rzp, eq=...) can be pinned in
// the case struct below (expected / expected_tol) to assert exact agreement.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>

#include "pcms/field/function_space/nimrod.h"
#include "pcms/field/evaluation_request.h"
#include "pcms/utility/arrays.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace
{

// Everything needed to drive one NIMROD evaluation case. Hardcoded per case;
// the test runs in the case's working directory (which holds nimrod.in).
struct NimrodEvalCase
{
  std::string dump;              // dump file (relative to working dir)
  std::string fieldlist;         // fields to initialize (NIMROD field ids)
  pcms::NimrodInputTraits traits; // sizing traits matching nimrod.in
  std::string field;             // field id to evaluate
  int eq;                        // equilibrium/perturbation selector
  std::array<double, 3> in_bounds;      // an in-bounds (R,Z,Phi) query point
  std::array<double, 3> out_of_bounds;  // a definitely out-of-bounds point
  std::vector<double> expected;  // nimpy reference values at in_bounds
};

// cylrwtm: LAR cylinder, circular toroidal cross-section centered at R=10.0,
// Z=0 with minor radius 1.4, poly_degree=3, gll nodes. Magnetic axis ~ (10,0).
const NimrodEvalCase kCylrwtm{
  /*dump=*/"dumpgll.00000.h5",
  /*fieldlist=*/"bvn",
  /*traits=*/pcms::NimrodInputTraits{/*nisp=*/0, /*nnsp=*/0, /*ds_nqty=*/1,
                                     /*zimp=*/0, /*nmodes=*/1},
  /*field=*/"b",
  /*eq=*/2,
  /*in_bounds=*/{10.0, 0.0, 0.0},
  /*out_of_bounds=*/{1.0e6, 1.0e6, 0.0},
  // nimpy EvalNimrod.eval_field('b', [10,0,0], dmode=0, eq=2):
  /*expected=*/{2.64781833e-10, 1.46768796e-10, 5.00000001}};

// Build a device coordinate view from host (R,Z,Phi) rows.
pcms::CoordinateView<pcms::DeviceMemorySpace> MakeCoords(
  Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace>& storage,
  const std::vector<std::array<double, 3>>& pts)
{
  storage = Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace>(
    "nimrod_test_coords", pts.size(), 3);
  auto host = Kokkos::create_mirror_view(storage);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    host(i, 0) = pts[i][0];
    host(i, 1) = pts[i][1];
    host(i, 2) = pts[i][2];
  }
  Kokkos::deep_copy(storage, host);
  return pcms::CoordinateView<pcms::DeviceMemorySpace>(
    pcms::CoordinateSystem::Cylindrical, pcms::MakeConstRank2View(storage));
}

} // namespace

TEST_CASE("NIMROD field evaluation", "[nimrod]")
{
  const NimrodEvalCase& c = kCylrwtm;

  auto space =
    pcms::NimrodFunctionSpace::FromDump(c.dump, c.traits, c.fieldlist, c.eq);

  const int nqty = c.traits.NqtyForField(c.field[0]);

  std::vector<std::array<double, 3>> pts = {c.in_bounds, c.out_of_bounds};

  Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> coord_storage;
  auto coords = MakeCoords(coord_storage, pts);
  auto request = pcms::EvaluationRequest::FromCoordinates(coords);

  auto evaluator = space.CreatePointEvaluator<double>(request);
  auto field = space.MakeField(c.field);

  Kokkos::View<double**, pcms::DeviceMemorySpace> out("nimrod_out", pts.size(),
                                                      nqty);
  evaluator->Evaluate(field, pcms::MakeRank2View(out));

  auto host = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(host, out);

  for (int q = 0; q < nqty; ++q) {
    UNSCOPED_INFO("field '" << c.field << "' eq=" << c.eq << " comp " << q
                            << " at (" << c.in_bounds[0] << "," << c.in_bounds[1]
                            << "," << c.in_bounds[2] << ") = " << host(0, q));
  }

  SECTION("in-bounds evaluation is finite")
  {
    for (int q = 0; q < nqty; ++q) {
      CHECK(std::isfinite(host(0, q)));
    }
  }

  SECTION("in-bounds evaluation matches nimpy reference")
  {
    REQUIRE(static_cast<int>(c.expected.size()) == nqty);
    for (int q = 0; q < nqty; ++q) {
      // Same underlying library as nimpy, so agreement is to round-off.
      CHECK(host(0, q) ==
            Catch::Approx(c.expected[q]).epsilon(1e-6).margin(1e-15));
    }
  }

  SECTION("out-of-bounds evaluation is NaN")
  {
    for (int q = 0; q < nqty; ++q) {
      CHECK(std::isnan(host(1, q)));
    }
  }
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard guard(argc, argv);
    result = Catch::Session().run(argc, argv);
  }
  MPI_Finalize();
  return result;
}
