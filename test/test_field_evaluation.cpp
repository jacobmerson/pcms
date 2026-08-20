#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mesh.hpp>
#include "pcms/field/function_space/lagrange.h"
#include "pcms/field/field_metadata.h"
#include "pcms/utility/assert.h"
#include "field_test_utils.h"
#include <cmath>
#include "pcms/field/coordinate_systems/cartesian.hpp"

using pcms::Real;

// Non-linear test function used to exercise quadratic interpolation.
KOKKOS_INLINE_FUNCTION static Real sin_f(Real x, Real y)
{
  return std::sin(20 * x * y) / 2 + 0.5;
}

// Standard interior test points shared across all evaluation tests.
static const std::vector<Real> kEvalCoords = {
  0.7681, 0.886,  0.5337, 0.5205,   0.8088, 0.1513, 0.13,
  0.43,   0.5484, 0.8263, 0.006119, 0.8642, 0.5889, 0.5622,
  0.9268, 0.1749, 0.2615, 0.1468,   0.9793, 0.9612,
};

TEST_CASE("evaluate linear 2d omega_h_field")
{
  auto lib = Omega_h::Library{};
  auto mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 100,
                                 100, 0, false);
  auto factory = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, pcms::csys::Cartesian::Deferred());
  auto field = factory->CreateFunction<Real>();

  pcms::test::SetField(
    field.GetData(), *factory->GetLayout(),
    OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });
  pcms::test::CheckEvaluation(
    factory, field, pcms::test::StandardEvalCoords2D(),
    OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });
}

#ifdef PCMS_ENABLE_MESHFIELDS
TEST_CASE("evaluate quadratic 2d meshfields_field")
{
  auto lib = Omega_h::Library{};
  auto mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 100,
                                 100, 0, false);
  auto factory = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 2, 1, pcms::csys::Cartesian::Deferred(), "global",
    pcms::LagrangeFunctionSpace::Backend::MeshFields);

  // Quadratic DOF holders span vertices and edge midpoints; the layout's DOF
  // coordinates cover both, so SetField samples sin_f at every holder.
  auto field = factory->CreateFunction<Real>();
  pcms::test::SetField(
    field, OMEGA_H_LAMBDA(Real x, Real y) { return sin_f(x, y); });

  pcms::test::CheckEvaluation(
    factory, field, kEvalCoords,
    OMEGA_H_LAMBDA(Real x, Real y) { return std::sin(20 * x * y) / 2 + 0.5; },
    1.0e-2);
}
#endif

TEST_CASE("evaluate quadratic 2d omega_h_field throws")
{
  auto lib = Omega_h::Library{};
  auto mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 100,
                                 100, 0, false);

  REQUIRE_THROWS_AS(pcms::LagrangeFunctionSpace::FromMesh(
                      mesh, 2, 1, pcms::csys::Cartesian::Deferred(), "global",
                      pcms::LagrangeFunctionSpace::Backend::OmegaH),
                    pcms::pcms_error);
}
