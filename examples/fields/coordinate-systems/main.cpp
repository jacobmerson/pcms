// coordinate-systems: fields in cylindrical coordinates, queried in Cartesian.
//
// Simulation codes use different coordinates: a tokamak code works in
// (R, phi, Z), a particle code in (x, y, z). PCMS tags every set of
// coordinates with a CoordinateSystem, and every vector or tensor value with
// the basis its components are in, so that a field can be evaluated at points
// given in other coordinates and come out in the components the caller asks
// for.
//
// Two ideas, both shown here:
//   * the points: a CoordinateMap carries query points from the caller's
//     coordinate system to the geometry's, and the evaluation happens at the
//     mapped points;
//   * the values: a scalar is the same number in every system, but the
//     components of a vector are not; the map re-expresses them in the basis
//     of the query points (the radial component of a vector is not its x
//     component).
//
// The geometry is a box mesh whose coordinates are (r, theta, z): r in
// [0, 2], theta in [0, 1.6] rad, z in [0, 1.1]. A P1 field on it is linear in
// (r, theta, z). We query it at Cartesian points with CartesianToCylindrical.
//
// What you will see, in order:
//   1. the map on its own: mapped points, and values rotated to Cartesian;
//   2. a scalar field evaluated at Cartesian points (QueryPoints with a map);
//   3. a vector field whose components are rotated into the query's basis;
//   4. the same through an Interpolator onto a Cartesian mesh, which binds
//      the target's points once with Geometry::Bind and the map;
//   5. a field with cylindrical points but Cartesian components, for which
//      the map passes the values through and only maps the points.

#include <pcms/backends/omega_h/omega_h.hpp>
#include <pcms/field/coordinate_map.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/query_points.hpp>
#include <pcms/transfer/interpolator.hpp>
#include <pcms/utility/assert.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>

#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace
{
using pcms::Real;

constexpr Real RoundingTol = 1e-12;
// With Cartesian components stored on a (r, theta, z) mesh the field is not
// linear in the mesh coordinates, so a P1 field only approximates it (section
// 5; the mesh has 8 x 8 x 6 cells).
constexpr Real BorrowedBasisTol = 2e-2;

// Sets the DOF holders of a 3-component field from f(p0, p1, p2, out): the
// coordinates of the holder in the field's system, and the components.
template <typename F>
void FillVector(pcms::Field<Real>& field, F f)
{
  const auto holders = example::DofCoordinates(field);
  example::HostMatrix data("data", holders.extent(0), 3);
  for (std::size_t i = 0; i < holders.extent(0); ++i) {
    f(holders(i, 0), holders(i, 1), holders(i, 2), &data(i, 0));
  }
  example::SetDofData(field, data);
}

// The largest |values(i, d) - points(i, d)| over rows i and components d.
Real MaxDifference(const example::HostMatrix& values,
                   const example::HostMatrix& points)
{
  Real error = 0.0;
  for (std::size_t i = 0; i < points.extent(0); ++i) {
    for (int d = 0; d < 3; ++d) {
      error = std::fmax(error, std::fabs(values(i, d) - points(i, d)));
    }
  }
  return error;
}
} // namespace

int main(int argc, char** argv)
{
  Omega_h::Library lib(&argc, &argv);
  example::Report report;
  {
    const auto cartesian = pcms::csys::Cartesian::Create(3);
    const auto cylindrical = pcms::csys::CylindricalRThetaZ::Create();

    // The basis of a vector whose components are physical, the same unit in
    // every direction (not scaled by the coordinates), in a given system.
    const auto vector_in =
      [](std::shared_ptr<const pcms::CoordinateSystem> system) {
        return pcms::ValueBasis{std::move(system),
                                pcms::ComponentScaling::Physical,
                                pcms::values::Vector};
      };

    // -------------------------------------------------------------------
    // 1. A coordinate map on its own
    //
    // CartesianToCylindrical maps points from Cartesian (x, y, z) to
    // cylindrical (r, theta, z). Binding it to a set of points does all the
    // per-point work once: the mapped points, and what is needed to rotate
    // components at them. The values run the other way: it turns a vector
    // given in cylindrical components into Cartesian ones, in place.
    //
    // Take e_theta, the unit vector of increasing theta, at three points. Its
    // cylindrical components are (0, 1, 0) everywhere; in Cartesian
    // components it is (-sin theta, cos theta, 0).
    auto xyz =
      example::ToDevice({1.0, 0.0, 0.5, 0.0, 2.0, -1.0, 1.0, 1.0, 0.0}, 3);
    const auto bound =
      pcms::CartesianToCylindrical{}.Bind(example::Points(cartesian, xyz));
    const auto xyz_host = example::ToHost(xyz);
    const auto rtz = example::ToHost(bound->MappedPoints().GetValues());
    auto e_theta =
      example::ToDevice({0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0}, 3);
    const auto rotated =
      bound->TransformValues(example::Output(vector_in(cylindrical), e_theta));
    const auto e_cartesian = example::ToHost(e_theta);
    Real error = 0.0;
    for (int i = 0; i < 3; ++i) {
      const Real r = std::hypot(xyz_host(i, 0), xyz_host(i, 1));
      error = std::fmax(error, std::fabs(rtz(i, 0) - r));
      error =
        std::fmax(error, std::fabs(e_cartesian(i, 0) + std::sin(rtz(i, 1))));
      error =
        std::fmax(error, std::fabs(e_cartesian(i, 1) - std::cos(rtz(i, 1))));
      error = std::fmax(error, std::fabs(e_cartesian(i, 2)));
      std::printf("  (r, theta, z) = (%.3f, %.3f, %.3f), e_theta in Cartesian "
                  "components = (%.3f, %.3f, %.3f)\n",
                  rtz(i, 0), rtz(i, 1), rtz(i, 2), e_cartesian(i, 0),
                  e_cartesian(i, 1), e_cartesian(i, 2));
    }
    report.CheckAtMost("map: mapped points and rotated e_theta", error,
                       RoundingTol);
    // The result says which basis it is in: that of the query points.
    report.Check("map: the rotated values are tagged Cartesian",
                 rotated.GetBasis().system->Kind() == "cartesian" &&
                   rotated.GetBasis().variance == pcms::values::Vector,
                 1);

    // -------------------------------------------------------------------
    // 2. A scalar field at Cartesian points
    //
    // The source is the box in (r, theta, z), with P1 spaces of one and of
    // three components. QueryPoints takes the map as its third argument:
    // the points given to Locate are Cartesian, they are mapped to
    // (r, theta, z), located there, and the field is evaluated at the mapped
    // points. g = 1 + 2 r + 3 z is linear in the mesh's coordinates, so it is
    // reproduced exactly, though it is not linear in x and y.
    auto source_mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 2.0, 1.6, 1.1, 8, 8, 6);
    const auto source_geometry =
      pcms::backend::omega_h::CreateGeometry(source_mesh, cylindrical);
    const pcms::LagrangeOptions omega_h{.backend =
                                          pcms::LagrangeBackend::OmegaH};
    const auto scalars =
      pcms::CreateLagrangeSpace(source_geometry, 1, 1, omega_h);
    const auto vectors =
      pcms::CreateLagrangeSpace(source_geometry, 1, 3, omega_h);

    auto g = scalars->CreateFunction<Real>("g");
    {
      const auto holders = example::DofCoordinates(g);
      example::HostMatrix data("data", holders.extent(0), 1);
      for (std::size_t i = 0; i < holders.extent(0); ++i) {
        data(i, 0) = 1.0 + 2.0 * holders(i, 0) + 3.0 * holders(i, 2);
      }
      example::SetDofData(g, data);
    }

    // Points in the part of the cylinder the mesh covers: x, y > 0 keeps
    // theta between 0 and pi/2.
    std::mt19937 random(11);
    std::uniform_real_distribution<Real> xy(0.2, 1.0), z(0.1, 1.0);
    example::HostMatrix points("points", 100, 3);
    for (std::size_t i = 0; i < points.extent(0); ++i) {
      points(i, 0) = xy(random);
      points(i, 1) = xy(random);
      points(i, 2) = z(random);
    }
    const auto device_points = example::ToDevice(points);
    const auto to_cylindrical =
      std::make_shared<pcms::CartesianToCylindrical>();

    pcms::QueryPoints query(source_geometry, pcms::OutOfBoundsPolicy{},
                            to_cylindrical);
    query.Locate(example::Points(cartesian, device_points));
    example::DeviceMatrix scalar_out("scalar_out", points.extent(0), 1);
    query.Evaluate(g, example::Output(g.GetData().GetValueBasis(), scalar_out));
    const auto scalar_host = example::ToHost(scalar_out);
    error = 0.0;
    for (std::size_t i = 0; i < points.extent(0); ++i) {
      const Real r = std::hypot(points(i, 0), points(i, 1));
      error = std::fmax(error, std::fabs(scalar_host(i, 0) -
                                         (1.0 + 2.0 * r + 3.0 * points(i, 2))));
    }
    report.CheckAtMost("scalar at Cartesian points: g(r(x, y), z)", error,
                       RoundingTol);

    // -------------------------------------------------------------------
    // 3. A vector field, rotated into the basis of the query points
    //
    // B = r e_r + z e_z, stored in physical cylindrical components (r, 0, z)
    // at the DOF holders, is the vector (x, y, z) in Cartesian components.
    // Components are not the same in every system, so the field says which
    // basis its components are in: the default is that of the space, here
    // the cylindrical system of the mesh. The map knows how to turn them into
    // Cartesian components at each query point, and the output is declared to
    // be in that basis: asking for another one is an error, not a silent
    // mistake.
    auto b = vectors->CreateFunction<Real>("B", pcms::values::Vector,
                                           pcms::ComponentScaling::Physical);
    FillVector(b, [](Real r, Real, Real z, Real* out) {
      out[0] = r;
      out[1] = 0.0;
      out[2] = z;
    });
    example::DeviceMatrix vector_out("vector_out", points.extent(0), 3);
    query.Evaluate(b, example::Output(vector_in(cartesian), vector_out));
    report.CheckAtMost("vector at Cartesian points: B = (x, y, z)",
                       MaxDifference(example::ToHost(vector_out), points),
                       RoundingTol);
    bool refused = false;
    try {
      query.Evaluate(b, example::Output(vector_in(cylindrical), vector_out));
    } catch (const pcms::pcms_error&) {
      refused = true;
    }
    report.Check("vector: an output in another basis is refused", refused,
                 refused ? 1 : 0);

    // -------------------------------------------------------------------
    // 4. The same through an Interpolator
    //
    // To fill a whole field of another mesh with the values of this one, bind
    // the DOF holders of the target once with Geometry::Bind and the map,
    // and give the bound points to an Interpolator, which applies to as many
    // fields as you like without locating again. The target is the unit cube
    // in Cartesian coordinates, which lies inside the source box.
    auto target_mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 1.0, 4, 4, 4);
    const auto target = pcms::CreateLagrangeSpace(
      pcms::backend::omega_h::CreateGeometry(target_mesh, cartesian), 1, 3,
      omega_h);
    const auto bound_target =
      source_geometry->Bind(target->GetLayout()->GetDOFHolderCoordinates(),
                            pcms::OutOfBoundsPolicy{}, to_cylindrical);
    pcms::Interpolator<Real> interpolate(*vectors, *target, bound_target);

    auto b_target = target->CreateFunction<Real>("B", pcms::values::Vector);
    interpolate.Apply(b, b_target);
    report.CheckAtMost("interpolated onto a Cartesian mesh: B = (x, y, z)",
                       MaxDifference(example::DofData(b_target),
                                     example::DofCoordinates(b_target)),
                       RoundingTol);

    // -------------------------------------------------------------------
    // 5. Cylindrical points, Cartesian components
    //
    // The basis of the values does not have to be that of the points. A field
    // can have its points in (r, theta, z) and its components in Cartesian
    // x, y, z: say so when creating the function, by giving the system of the
    // components. B = (x, y, z) = (r cos theta, r sin theta, z) is then the
    // data. The values are already in the basis of the target, so the map
    // leaves them alone and only maps the points. The field is not linear in
    // (r, theta, z), so the P1 field only approximates it.
    auto b_borrowed =
      vectors->CreateFunction<Real>("B", pcms::values::Vector, cartesian);
    FillVector(b_borrowed, [](Real r, Real theta, Real z, Real* out) {
      out[0] = r * std::cos(theta);
      out[1] = r * std::sin(theta);
      out[2] = z;
    });
    auto b_borrowed_target =
      target->CreateFunction<Real>("B", pcms::values::Vector);
    interpolate.Apply(b_borrowed, b_borrowed_target);
    const Real borrowed_error =
      MaxDifference(example::DofData(b_borrowed_target),
                    example::DofCoordinates(b_borrowed_target));
    std::printf("Cartesian components on cylindrical points: error %.3e\n",
                borrowed_error);
    report.CheckAtMost("borrowed basis: B = (x, y, z) up to interpolation",
                       borrowed_error, BorrowedBasisTol);
  }
  return report.Finish();
}
