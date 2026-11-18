// moving-points: tracing field lines with a batch of tracers.
//
// Many codes evaluate a field at points that change at every step: particles
// pushed in a field, field lines followed through a magnetic field, probes
// that sweep a mesh. Locating a point on a mesh is the expensive part of
// evaluating a field there, so PCMS separates it: QueryPoints locates the
// points, keeps the result, and reuses its storage when you locate the next
// set of points.
//
// Here a batch of tracers follows the field lines of a rotation field
//     v(x, y) = (-(y - 1/2), x - 1/2)
// on the unit square, stored as a P1 vector field on a triangle mesh. The
// field lines are circles around the center, traversed once in 2 pi. Each
// tracer is advanced by the classical Runge-Kutta method (RK4), which
// evaluates the field at four sets of points per step: the tracers advanced
// by the previous stage.
//
// Tracers on small circles stay in the domain and come back to where they
// started. Tracers on a large circle leave the domain, where the field is
// not defined: with the FILL policy their points are reported as missing, and
// we stop advancing them.
//
// What you will see, in order:
//   1. the mesh, the vector field and the tracers;
//   2. one RK4 stage: Locate, then Evaluate, then read MissingPoints;
//   3. the integration loop over the stages;
//   4. the checks: closed circles inside, tracers lost outside.

#include <pcms/backends/omega_h/omega_h.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>

#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr pcms::Real Pi = 3.14159265358979323846;
constexpr pcms::Real Center = 0.5;

// RK4 with 200 steps per revolution has a global error of about
// (2 pi / 200)^4 / 120 of the radius per revolution, far below this.
constexpr pcms::Real ClosureTol = 1e-6;
constexpr int StepsPerRevolution = 200;

struct Ring
{
  pcms::Real radius;
  int num_tracers;
};

// Advances the tracers of a batch through a field. Each call of Velocity is
// one stage of RK4.
class Tracers
{
public:
  // `start` is copied: Kokkos views share their memory when copied.
  Tracers(std::shared_ptr<const pcms::Geometry> geometry,
          std::shared_ptr<const pcms::CoordinateSystem> system,
          const pcms::Function<pcms::Real>& velocity,
          const example::HostMatrix& start)
    : position("position", start.extent(0), 2),
      lost(start.extent(0), false),
      system_(std::move(system)),
      velocity_(velocity),
      // Points outside the domain get the velocity 0 instead of an error.
      query_(std::move(geometry),
             pcms::OutOfBoundsPolicy{.mode = pcms::OutOfBoundsMode::FILL,
                                     .fill_value = 0.0})
  {
    Kokkos::deep_copy(position, start);
  }

  // The velocity at `points`: Locate finds the element of every point and
  // Evaluate interpolates the vertex values in it. QueryPoints::Locate
  // keeps the storage of the previous call when the number of points does not
  // grow, so the loop below allocates nothing in steady state. Points outside
  // the mesh are not evaluated: their rows of the result hold the fill value,
  // and MissingPoints lists them.
  example::HostMatrix Velocity(const example::HostMatrix& points)
  {
    query_.Locate(example::Points(system_, example::ToDevice(points)));
    example::DeviceMatrix out("velocity", points.extent(0), 2);
    query_.Evaluate(velocity_,
                    example::Output(velocity_.GetData().GetValueBasis(), out));
    const auto missing = Kokkos::create_mirror_view_and_copy(
      pcms::HostMemorySpace(), query_.Bound()->MissingPoints());
    num_missing_last = static_cast<int>(missing.extent(0));
    for (std::size_t k = 0; k < missing.extent(0); ++k) {
      lost[missing(k)] = true;
    }
    return example::ToHost(out);
  }

  // One RK4 step of size dt for the tracers that are not lost.
  void Step(pcms::Real dt)
  {
    const std::size_t n = position.extent(0);
    auto stage_point = [&](const example::HostMatrix& k, pcms::Real h) {
      example::HostMatrix p("stage", n, 2);
      for (std::size_t i = 0; i < n; ++i) {
        for (int d = 0; d < 2; ++d) {
          p(i, d) = position(i, d) + h * k(i, d);
        }
      }
      return p;
    };
    const auto k1 = Velocity(position);
    const auto k2 = Velocity(stage_point(k1, 0.5 * dt));
    const auto k3 = Velocity(stage_point(k2, 0.5 * dt));
    const auto k4 = Velocity(stage_point(k3, dt));
    for (std::size_t i = 0; i < n; ++i) {
      if (lost[i]) {
        continue; // the tracer left the domain: it stays where it was
      }
      for (int d = 0; d < 2; ++d) {
        position(i, d) +=
          dt / 6.0 * (k1(i, d) + 2.0 * k2(i, d) + 2.0 * k3(i, d) + k4(i, d));
      }
    }
  }

  example::HostMatrix position;
  std::vector<bool> lost;
  int num_missing_last = 0;

private:
  std::shared_ptr<const pcms::CoordinateSystem> system_;
  const pcms::Function<pcms::Real>& velocity_;
  pcms::QueryPoints query_;
};
} // namespace

int main(int argc, char** argv)
{
  Omega_h::Library lib(&argc, &argv);
  example::Report report;
  {
    // -------------------------------------------------------------------
    // 1. Mesh, vector field and tracers
    //
    // The field has two components, so it is a vector: the values are tagged
    // as a vector in the Cartesian system of the mesh, which tells PCMS how
    // the components change when the field is re-expressed in other
    // coordinates (see coordinate-systems). Its values at the vertices are
    // set from v; a linear field is exact on a P1 mesh.
    auto mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0,
                                   16, 16, 0);
    const auto cartesian = pcms::csys::Cartesian::Create(2);
    const auto geometry =
      pcms::backend::omega_h::CreateGeometry(mesh, cartesian);
    const auto space = pcms::CreateLagrangeSpace(
      geometry, 1, 2,
      pcms::LagrangeOptions{.backend = pcms::LagrangeBackend::OmegaH});
    auto velocity =
      space->CreateFunction<pcms::Real>("velocity", pcms::values::Vector);
    const auto holders = example::DofCoordinates(velocity);
    example::HostMatrix data("data", holders.extent(0), 2);
    for (std::size_t i = 0; i < holders.extent(0); ++i) {
      data(i, 0) = -(holders(i, 1) - Center);
      data(i, 1) = holders(i, 0) - Center;
    }
    example::SetDofData(velocity, data);

    // Tracers start on circles around the center: three small ones, which
    // stay in the unit square, and a large one, which does not.
    const std::vector<Ring> rings{{0.15, 8}, {0.25, 8}, {0.35, 8}, {0.60, 12}};
    int num_tracers = 0;
    for (const auto& ring : rings) {
      num_tracers += ring.num_tracers;
    }
    example::HostMatrix start("start", num_tracers, 2);
    std::vector<int> ring_of(num_tracers);
    for (int i = 0, ring_index = 0; ring_index < static_cast<int>(rings.size());
         ++ring_index) {
      for (int k = 0; k < rings[ring_index].num_tracers; ++k, ++i) {
        const pcms::Real angle =
          0.1 + 2.0 * Pi * k / rings[ring_index].num_tracers;
        start(i, 0) = Center + rings[ring_index].radius * std::cos(angle);
        start(i, 1) = Center + rings[ring_index].radius * std::sin(angle);
        ring_of[i] = ring_index;
      }
    }
    Tracers tracers(geometry, cartesian, velocity, start);

    // -------------------------------------------------------------------
    // 2. One stage: found and missing points
    //
    // The first stage evaluates the field at the starting points. The large
    // circle has 12 tracers; those that start outside the unit square are
    // missing. We compare with the answer from the geometry of the circle.
    tracers.Velocity(tracers.position);
    int outside = 0;
    for (int i = 0; i < num_tracers; ++i) {
      const bool in_square = std::fabs(start(i, 0) - Center) <= 0.5 &&
                             std::fabs(start(i, 1) - Center) <= 0.5;
      outside += in_square ? 0 : 1;
    }
    std::printf("%d tracers, %d of them start outside the unit square "
                "(%d missing points)\n",
                num_tracers, outside, tracers.num_missing_last);
    report.Check("first stage: MissingPoints are the tracers outside the mesh",
                 tracers.num_missing_last == outside, tracers.num_missing_last);
    // Tracers that start outside have nothing to trace: they are lost from
    // the beginning. The others must not be marked by this stage.
    int lost_at_start = 0;
    for (int i = 0; i < num_tracers; ++i) {
      lost_at_start += tracers.lost[i] ? 1 : 0;
    }
    report.Check("first stage: lost tracers are those that start outside",
                 lost_at_start == outside, lost_at_start);

    // -------------------------------------------------------------------
    // 3. The integration loop
    //
    // One revolution takes 2 pi (the angular velocity is 1). A tracer that is
    // lost in a stage, because the stage's point is outside the mesh, stops.
    const pcms::Real dt = 2.0 * Pi / StepsPerRevolution;
    for (int step = 0; step < StepsPerRevolution; ++step) {
      tracers.Step(dt);
    }

    // -------------------------------------------------------------------
    // 4. The checks
    //
    // The tracers inside come back to their start, on their circle. Those on
    // the large circle all leave the domain, so all are lost.
    pcms::Real closure = 0.0, radial = 0.0;
    int lost_inside = 0, lost_outside = 0, num_outside_ring = 0;
    for (int i = 0; i < num_tracers; ++i) {
      if (ring_of[i] == static_cast<int>(rings.size()) - 1) {
        ++num_outside_ring;
        lost_outside += tracers.lost[i] ? 1 : 0;
        continue;
      }
      lost_inside += tracers.lost[i] ? 1 : 0;
      const pcms::Real dx = tracers.position(i, 0) - start(i, 0);
      const pcms::Real dy = tracers.position(i, 1) - start(i, 1);
      closure = std::fmax(closure, std::hypot(dx, dy));
      const pcms::Real r = std::hypot(tracers.position(i, 0) - Center,
                                      tracers.position(i, 1) - Center);
      radial = std::fmax(radial, std::fabs(r - rings[ring_of[i]].radius));
    }
    std::printf("after one revolution: closure error %.3e, radius error "
                "%.3e\n",
                closure, radial);
    report.Check("circles inside the square: no tracer is lost",
                 lost_inside == 0, lost_inside);
    report.CheckAtMost("circles inside the square: back at the start", closure,
                       ClosureTol);
    report.CheckAtMost("circles inside the square: radius conserved", radial,
                       ClosureTol);
    report.Check("circle outside the square: every tracer is lost",
                 lost_outside == num_outside_ring, lost_outside);
  }
  return report.Finish();
}
