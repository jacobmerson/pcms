// mesh-fields: Lagrange fields on an Omega_h mesh.
//
// hello-field evaluated a field on a grid. Here the geometry is an
// unstructured simplex mesh, built with Omega_h, and we look at what a field
// on it consists of: where its DOF holders are, which of them this process
// owns, how they are numbered across processes, and what the space's order
// changes (P0, P1 and, with MeshFields, P2).
//
// What you will see, in order:
//   1. an Omega_h box mesh, its geometry, and P0 and P1 spaces on it;
//   2. the layout of a field: owned and ghost DOF holders, global ids;
//   3. fields from functions, evaluated at points: P1 reproduces a linear
//      function exactly, P0 (one value per element) does not;
//   4. which differential operators a space supports;
//   5. P2, a quadratic space, through MeshFields when PCMS has it.

#include <pcms/backends/omega_h/omega_h.hpp>
#include <pcms/configuration.hpp>
#include <pcms/field/differential_operator.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>

#include <Kokkos_Core.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace
{
// f is linear: P1 contains it, P0 does not.
pcms::Real Linear(pcms::Real x, pcms::Real y)
{
  return 1.0 + 2.0 * x - 3.0 * y;
}
constexpr pcms::Real GradientNorm = 3.6055512754639896; // sqrt(2^2 + 3^2)

// f is quadratic: P2 contains it, P1 does not.
pcms::Real Quadratic(pcms::Real x, pcms::Real y)
{
  return 1.0 + x * x + x * y - 2.0 * y * y;
}

constexpr pcms::Real RoundingTol = 1e-12;

// Sets the DOF holders of `field` to f at their coordinates.
template <typename F>
void SetFromFunction(pcms::Field<pcms::Real>& field, F f)
{
  const auto holders = example::DofCoordinates(field);
  example::HostMatrix values("values", holders.extent(0), 1);
  for (std::size_t i = 0; i < holders.extent(0); ++i) {
    values(i, 0) = f(holders(i, 0), holders(i, 1));
  }
  example::SetDofData(field, values);
}

// The largest difference between the field at the points and f. The points
// are rows of `points`, in the coordinate system `system`.
template <typename F>
pcms::Real MaxError(const pcms::Function<pcms::Real>& field,
                    const std::shared_ptr<const pcms::CoordinateSystem>& system,
                    const example::HostMatrix& points, F f)
{
  const auto device_points =
    Kokkos::create_mirror_view_and_copy(pcms::DeviceMemorySpace(), points);
  example::DeviceMatrix out("out", points.extent(0), 1);
  pcms::Evaluate(field, example::Points(system, device_points),
                 example::Output(field.GetData().GetValueBasis(), out));
  const auto result = example::ToHost(out);
  pcms::Real error = 0.0;
  for (std::size_t i = 0; i < points.extent(0); ++i) {
    error =
      std::fmax(error, std::fabs(result(i, 0) - f(points(i, 0), points(i, 1))));
  }
  return error;
}

// Prints what the layout of a space says about its DOF holders.
void Describe(const char* name, const pcms::FieldLayout& layout,
              example::Report& report, long expected_global, bool serial)
{
  const auto owned = layout.GetOwnedHost();
  const auto gids = layout.GetGidsHost();
  long num_owned_flags = 0;
  std::set<pcms::GO> distinct;
  for (std::size_t i = 0; i < owned.extent(0); ++i) {
    num_owned_flags += owned(i) ? 1 : 0;
    distinct.insert(gids(i));
  }
  std::printf("%s: %d owned DOF holders of %ld in all, %d held in all, "
              "distributed layout %s, ghosts %s; global ids",
              name, static_cast<int>(layout.GetNumDofHolder()),
              static_cast<long>(layout.GetNumGlobalDofHolder()),
              static_cast<int>(owned.extent(0)),
              layout.IsDistributed() ? "yes" : "no",
              layout.HasGhosts() ? "yes" : "no");
  for (std::size_t i = 0; i < 4 && i < gids.extent(0); ++i) {
    std::printf(" %ld", static_cast<long>(gids(i)));
  }
  std::printf(" ...\n");

  // GetNumDofHolder counts the DOF holders this process owns; the owned flags
  // cover all it holds, owned or ghost, so their count of owned holders must
  // agree. On one process there are no ghosts, and the global ids are the
  // numbers 0 .. n-1 of the mesh entities, each once.
  report.Check(std::string(name) + ": owned flags agree with the count",
               num_owned_flags == layout.GetNumDofHolder(), num_owned_flags);
  if (serial) {
    report.Check(std::string(name) + ": every DOF holder is owned, none ghost",
                 !layout.HasGhosts() &&
                   owned.extent(0) ==
                     static_cast<std::size_t>(layout.GetNumDofHolder()),
                 static_cast<long>(owned.extent(0)));
    report.Check(std::string(name) + ": global ids are distinct",
                 distinct.size() == gids.extent(0),
                 static_cast<long>(distinct.size()));
    report.Check(std::string(name) + ": one DOF holder per entity",
                 layout.GetNumGlobalDofHolder() == expected_global,
                 static_cast<long>(layout.GetNumGlobalDofHolder()));
  }
}
} // namespace

int main(int argc, char** argv)
{
  // Omega_h starts MPI and Kokkos, which PCMS uses, and shuts them down when
  // the library object goes away.
  Omega_h::Library lib(&argc, &argv);
  example::Report report;
  {
    // -------------------------------------------------------------------
    // 1. Mesh, geometry and spaces
    //
    // A 2D box of 8 x 8 squares, each split into two triangles: 81 vertices
    // and 128 elements. The geometry of the mesh builds the search that finds
    // the element around a point; it is expensive, so one geometry is made
    // per mesh and shared by all spaces on the mesh. The mesh must outlive
    // the geometry.
    //
    // Two Lagrange spaces: P0 has one DOF holder per element (its centroid),
    // P1 one per vertex. LagrangeBackend::OmegaH selects the Omega_h
    // implementation, which has orders 0 and 1; the MeshFields backend has
    // orders 1 and 2 (see the end).
    auto mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, 8, 8, 0);
    const auto cartesian = pcms::csys::Cartesian::Create(2);
    const auto geometry =
      pcms::backend::omega_h::CreateGeometry(mesh, cartesian);
    const pcms::LagrangeOptions omega_h{.backend =
                                          pcms::LagrangeBackend::OmegaH};
    const auto p0 = pcms::CreateLagrangeSpace(geometry, 0, 1, omega_h);
    const auto p1 = pcms::CreateLagrangeSpace(geometry, 1, 1, omega_h);

    // -------------------------------------------------------------------
    // 2. The layout of a field
    //
    // The layout of a space says how many DOF holders a field has, where
    // they are and how they are numbered. When a mesh is partitioned over
    // several processes, each process holds the DOF holders of its part of
    // the mesh. A DOF holder on the boundary between two parts exists on both
    // sides, but only one process owns it: the other holds a ghost copy
    // that Field::Synchronize refreshes from the owner. The numbers of a
    // field are about the owned DOF holders: GetNumDofHolder counts them, and
    // the values you set and get are those of the owned ones, in their order.
    // The global id of a DOF holder is the same on all processes that have
    // it; it names the holder across the whole mesh. GetOwnedHost and
    // GetGidsHost describe all the holders a process has, ghosts included.
    // With one process there are no ghosts, though the layout is still a
    // distributed one: IsDistributed says that it can be partitioned.
    const bool serial = lib.world()->size() == 1;
    std::printf("%d process(es)\n", static_cast<int>(lib.world()->size()));
    Describe("P0 on elements", *p0->GetLayout(), report, mesh.nelems(), serial);
    Describe("P1 on vertices", *p1->GetLayout(), report, mesh.nverts(), serial);

    // -------------------------------------------------------------------
    // 3. Fields from functions, evaluated at points
    //
    // The same linear function in both spaces. Evaluate gives the value of
    // the field at any point: for P1 that is the linear interpolation of the
    // vertex values of the element the point is in, which is exact for a
    // linear function. For P0 it is the value of the element, that is f at
    // the centroid: exact at the centroids, off by up to |grad f| times the
    // distance to the centroid elsewhere.
    auto f1 = p1->CreateFunction<pcms::Real>("f1");
    auto f0 = p0->CreateFunction<pcms::Real>("f0");
    SetFromFunction(f1, Linear);
    SetFromFunction(f0, Linear);

    std::mt19937 random(7);
    std::uniform_real_distribution<pcms::Real> unit(0.0, 1.0);
    example::HostMatrix points("points", 200, 2);
    for (std::size_t i = 0; i < points.extent(0); ++i) {
      points(i, 0) = unit(random);
      points(i, 1) = unit(random);
    }
    report.CheckAtMost("P1: error of a linear field at random points",
                       MaxError(f1, cartesian, points, Linear), RoundingTol);

    const auto centroids = example::DofCoordinates(f0);
    report.CheckAtMost("P0: error at its own DOF holders (the centroids)",
                       MaxError(f0, cartesian, centroids, Linear), RoundingTol);
    const pcms::Real element_size = std::sqrt(2.0) / 8.0; // longest edge
    const pcms::Real p0_error = MaxError(f0, cartesian, points, Linear);
    std::printf("P0 error at random points: %.3e (the bound is %.3e)\n",
                p0_error, GradientNorm * element_size);
    report.CheckAtMost("P0: error at random points, at most |grad f| h",
                       p0_error, GradientNorm * element_size);

    // -------------------------------------------------------------------
    // 4. What a space can compute
    //
    // A space says which differential operators its evaluators support.
    // Value is supported by all; the Lagrange spaces of this release do not
    // support Gradient yet (the spline space of splines-eqdsk does).
    for (const auto& [name, space] :
         {std::pair{"P0", p0}, std::pair{"P1", p1}}) {
      std::printf(
        "%s supports Value: %s, Gradient: %s\n", name,
        space->Supports(pcms::DifferentialOperator::Value) ? "yes" : "no",
        space->Supports(pcms::DifferentialOperator::Gradient) ? "yes" : "no");
    }
    report.Check("P1 supports evaluating values",
                 p1->Supports(pcms::DifferentialOperator::Value), 1);

    // -------------------------------------------------------------------
    // 5. P2 through MeshFields
    //
    // Quadratic Lagrange fields have DOF holders at the vertices and at the
    // edge midpoints. They come from the MeshFields backend, which is
    // optional; the program checks the build option the way a user's program
    // does, with the macro of pcms/configuration.hpp.
#ifdef PCMS_ENABLE_MESHFIELDS
    const auto p2 = pcms::CreateLagrangeSpace(
      geometry, 2, 1,
      pcms::LagrangeOptions{.backend = pcms::LagrangeBackend::MeshFields});
    auto q2 = p2->CreateFunction<pcms::Real>("q2");
    SetFromFunction(q2, Quadratic);
    std::printf("P2: %d DOF holders (%d vertices and %d edges)\n",
                static_cast<int>(q2.GetLayout().GetNumDofHolder()),
                static_cast<int>(mesh.nverts()),
                static_cast<int>(mesh.nedges()));
    report.Check("P2: one DOF holder per vertex and per edge",
                 q2.GetLayout().GetNumDofHolder() ==
                   mesh.nverts() + mesh.nedges(),
                 static_cast<long>(q2.GetLayout().GetNumDofHolder()));
    report.CheckAtMost("P2: error of a quadratic field at random points",
                       MaxError(q2, cartesian, points, Quadratic), 1e-10);
    // A linear field cannot represent the quadratic, so P1 is off.
    auto q1 = p1->CreateFunction<pcms::Real>("q1");
    SetFromFunction(q1, Quadratic);
    const pcms::Real p1_error = MaxError(q1, cartesian, points, Quadratic);
    std::printf("P1 error for the quadratic function: %.3e\n", p1_error);
    report.Check("P1 cannot represent a quadratic function", p1_error > 1e-3,
                 1);
#else
    (void)Quadratic;
    std::printf("PCMS was built without MeshFields: P2 is skipped\n");
#endif
  }
  return report.Finish();
}
