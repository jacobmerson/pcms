#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_library.hpp>
#include <pcms/field/coordinate_map.hpp>
#include <pcms/field/coordinate_systems/cylindrical.hpp>
#include <pcms/field/evaluator/host_point_evaluator.hpp>
#include <pcms/field/external_source.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/transformed_source.hpp>
#include <pcms/transfer/interpolator.h>
#include "field_test_utils.h"
#include <array>
#include <cmath>
#include <memory>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pcms::ComponentScaling;
using pcms::CoordinateView;
using pcms::DeviceMemorySpace;
using pcms::EvaluationRequest;
using pcms::Field;
using pcms::HostMemorySpace;
using pcms::LO;
using pcms::OutOfBoundsMode;
using pcms::OutOfBoundsPolicy;
using pcms::Real;
using pcms::ValueBasis;
using pcms::ValueView;
namespace values = pcms::values;

namespace
{

constexpr double kTol = 1e-12;

// Quantities of the mock source; each is an analytic function of
// (R, theta, Z) on the domain 0 <= R <= kMaxR.
enum class MockQuantity
{
  Scalar,     // f = 2R + 3 theta - Z
  Vector,     // (R, R theta, Z) in physical cylindrical components
  TwoScalars, // (R + Z, R - Z): a multi-component scalar
};

constexpr Real kMaxR = 1.5;

std::array<Real, 3> MockValues(MockQuantity q, const Real* x)
{
  const Real r = x[0], theta = x[1], z = x[2];
  switch (q) {
    case MockQuantity::Scalar: return {2.0 * r + 3.0 * theta - z, 0.0, 0.0};
    case MockQuantity::Vector: return {r, r * theta, z};
    case MockQuantity::TwoScalars: return {r + z, r - z, 0.0};
  }
  return {};
}

class MockSource;

struct MockHandle
{
  const MockSource* source;
  MockQuantity quantity;
};

// Counts backend calls so tests can check that points are located once, at
// construction, and evaluated only when inside the domain.
struct MockCounters
{
  int locate = 0;
  int evaluate = 0;
};

class MockBackend
{
public:
  MockBackend(const MockSource* source, std::shared_ptr<MockCounters> counters)
    : source_(source), counters_(std::move(counters))
  {
  }

  bool Locate(LO /*i*/, const Real* x)
  {
    ++counters_->locate;
    return x[0] >= 0.0 && x[0] <= kMaxR;
  }

  const MockHandle& Resolve(const Field<Real>& field) const
  {
    const auto* data =
      dynamic_cast<const pcms::HandleFieldData<Real, MockHandle>*>(
        &field.GetData());
    if (data == nullptr || data->GetHandle().source != source_) {
      throw pcms::pcms_error("MockSource: field is not a quantity of this "
                             "source");
    }
    return data->GetHandle();
  }

  bool EvaluatePoint(const MockHandle& handle, LO /*i*/, const Real* x,
                     Real* out)
  {
    ++counters_->evaluate;
    const auto v = MockValues(handle.quantity, x);
    const int n = handle.quantity == MockQuantity::Scalar   ? 1
                  : handle.quantity == MockQuantity::Vector ? 3
                                                            : 2;
    for (int c = 0; c < n; ++c) {
      out[c] = v[c];
    }
    return true;
  }

private:
  const MockSource* source_;
  std::shared_ptr<MockCounters> counters_;
};

class MockSource final : public pcms::ExternalSource
{
public:
  static std::shared_ptr<MockSource> Create()
  {
    return std::shared_ptr<MockSource>(new MockSource());
  }

  Field<Real> MakeScalar() const
  {
    return MakeField<Real>("f", MockHandle{this, MockQuantity::Scalar},
                           ValueBasis{}, 1);
  }
  Field<Real> MakeVector() const
  {
    return MakeField<Real>("v", MockHandle{this, MockQuantity::Vector},
                           CylindricalVector(), 3);
  }
  Field<Real> MakeTwoScalars() const
  {
    return MakeField<Real>("s2", MockHandle{this, MockQuantity::TwoScalars},
                           ValueBasis{}, 2);
  }

  static ValueBasis CylindricalVector()
  {
    return ValueBasis{pcms::csys::CylindricalRThetaZ::Create(),
                      ComponentScaling::Physical, values::Vector};
  }

  std::shared_ptr<MockCounters> counters = std::make_shared<MockCounters>();

protected:
  pcms::PointEvaluatorVariant CreatePointEvaluatorImpl(
    pcms::Type value_type, const EvaluationRequest& request) const override
  {
    if (value_type != pcms::Type::Real) {
      throw pcms::pcms_error("MockSource: only Real evaluation is supported");
    }
    return std::unique_ptr<pcms::PointEvaluator<Real>>(
      std::make_unique<pcms::HostPointEvaluator<MockBackend>>(
        MockBackend(this, counters), request, "MockSource"));
  }

private:
  MockSource()
    : ExternalSource(pcms::csys::CylindricalRThetaZ::Create(), "MockSource")
  {
  }
};

// Owns the device storage behind a CoordinateView built from host rows.
struct Points
{
  Points(std::shared_ptr<const pcms::CoordinateSystem> system_in,
         const std::vector<std::vector<Real>>& rows)
    : system(std::move(system_in)),
      storage("points", rows.size(), rows.front().size())
  {
    auto host = Kokkos::create_mirror_view(storage);
    for (size_t i = 0; i < rows.size(); ++i) {
      for (size_t d = 0; d < rows[i].size(); ++d) {
        host(i, d) = rows[i][d];
      }
    }
    Kokkos::deep_copy(storage, host);
  }

  CoordinateView<DeviceMemorySpace> View() const
  {
    return CoordinateView<DeviceMemorySpace>(system,
                                             pcms::MakeConstRank2View(storage));
  }

  std::shared_ptr<const pcms::CoordinateSystem> system;
  Kokkos::View<Real**, DeviceMemorySpace> storage;
};

Kokkos::View<Real**, HostMemorySpace> EvaluateToHost(
  const pcms::PointEvaluator<Real>& evaluator, const Field<Real>& field,
  const ValueBasis& tag, size_t n, size_t width)
{
  Kokkos::View<Real**, DeviceMemorySpace> out("out", n, width);
  evaluator.Evaluate(
    field, ValueView<Real, DeviceMemorySpace>(tag, pcms::MakeRank2View(out)));
  Kokkos::View<Real**, HostMemorySpace> host("host", n, width);
  pcms::DeepCopyMismatchLayouts(host, out);
  return host;
}

std::vector<LO> FilledToHost(const pcms::PointEvaluator<Real>& evaluator)
{
  const auto filled = evaluator.FilledPoints();
  auto host = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), filled);
  return std::vector<LO>(host.data(), host.data() + host.extent(0));
}

const std::vector<std::vector<Real>> kMixedPoints = {{0.5, 0.1, -0.2},
                                                     {2.0, 0.0, 0.0},
                                                     {1.0, -0.4, 0.3},
                                                     {-1.0, 0.0, 0.0}};

} // namespace

TEST_CASE("ExternalSource: FILL evaluates located points and fills the rest")
{
  auto source = MockSource::Create();
  Points pts(source->GetCoordinateSystem(), kMixedPoints);
  const Real fill = -7.0;
  auto evaluator =
    source->CreatePointEvaluator<Real>(EvaluationRequest::FromCoordinates(
      pts.View(), {OutOfBoundsMode::FILL, fill}));
  REQUIRE(FilledToHost(*evaluator) == std::vector<LO>{1, 3});

  auto f = source->MakeScalar();
  const auto result = EvaluateToHost(*evaluator, f, ValueBasis{}, 4, 1);
  for (int i : {0, 2}) {
    CAPTURE(i);
    const auto expected =
      MockValues(MockQuantity::Scalar, kMixedPoints[i].data());
    REQUIRE_THAT(result(i, 0), WithinAbs(expected[0], kTol));
  }
  REQUIRE(result(1, 0) == fill);
  REQUIRE(result(3, 0) == fill);
}

TEST_CASE("ExternalSource: points are located once, at construction")
{
  auto source = MockSource::Create();
  Points pts(source->GetCoordinateSystem(), kMixedPoints);
  auto evaluator = source->CreatePointEvaluator<Real>(
    EvaluationRequest::FromCoordinates(pts.View(), {OutOfBoundsMode::FILL}));
  REQUIRE(source->counters->locate == 4);

  auto f = source->MakeScalar();
  auto s2 = source->MakeTwoScalars();
  EvaluateToHost(*evaluator, f, ValueBasis{}, 4, 1);
  EvaluateToHost(*evaluator, s2, ValueBasis{}, 4, 2);
  REQUIRE(source->counters->locate == 4);
  // Only the two located points reach the backend, once per field.
  REQUIRE(source->counters->evaluate == 4);
}

TEST_CASE("ExternalSource: ERROR rejects points outside the domain")
{
  auto source = MockSource::Create();
  Points mixed(source->GetCoordinateSystem(), kMixedPoints);
  REQUIRE_THROWS_WITH(source->CreatePointEvaluator<Real>(
                        EvaluationRequest::FromCoordinates(mixed.View())),
                      ContainsSubstring("2 query point(s) are outside"));

  Points inside(source->GetCoordinateSystem(), {{0.5, 0.0, 0.0}});
  auto evaluator = source->CreatePointEvaluator<Real>(
    EvaluationRequest::FromCoordinates(inside.View()));
  REQUIRE(evaluator->FilledPoints().extent(0) == 0);
}

TEST_CASE("ExternalSource: NEAREST_BOUNDARY is not supported")
{
  auto source = MockSource::Create();
  Points pts(source->GetCoordinateSystem(), {{0.5, 0.0, 0.0}});
  REQUIRE_THROWS_WITH(
    source->CreatePointEvaluator<Real>(EvaluationRequest::FromCoordinates(
      pts.View(), {OutOfBoundsMode::NEAREST_BOUNDARY})),
    ContainsSubstring("nearest-boundary"));
}

TEST_CASE("ExternalSource: vector and multi-component scalar fields")
{
  auto source = MockSource::Create();
  const std::vector<std::vector<Real>> rows = {{0.5, 0.1, -0.2},
                                               {1.0, 0.3, 2.0}};
  Points pts(source->GetCoordinateSystem(), rows);
  auto evaluator = source->CreatePointEvaluator<Real>(
    EvaluationRequest::FromCoordinates(pts.View()));

  auto v = source->MakeVector();
  REQUIRE(v.GetName() == "v");
  REQUIRE(v.GetLayout().GetNumComponents() == 3);
  REQUIRE(v.GetLayout().GetNumOwnedDofHolder() == 0);
  REQUIRE(v.GetDOFHolderData().GetValues().extent(1) == 3);

  const auto vector_result =
    EvaluateToHost(*evaluator, v, MockSource::CylindricalVector(), 2, 3);
  auto s2 = source->MakeTwoScalars();
  const auto scalar_result = EvaluateToHost(*evaluator, s2, ValueBasis{}, 2, 2);
  for (int i = 0; i < 2; ++i) {
    CAPTURE(i);
    const Real* x = rows[i].data();
    const auto ev = MockValues(MockQuantity::Vector, x);
    const auto es = MockValues(MockQuantity::TwoScalars, x);
    for (int c = 0; c < 3; ++c) {
      REQUIRE_THAT(vector_result(i, c), WithinAbs(ev[c], kTol));
    }
    for (int c = 0; c < 2; ++c) {
      REQUIRE_THAT(scalar_result(i, c), WithinAbs(es[c], kTol));
    }
  }
}

TEST_CASE("ExternalSource: fields are evaluation only")
{
  auto source = MockSource::Create();
  auto f = source->MakeScalar();
  Kokkos::View<Real**, DeviceMemorySpace> data("data", 0, 1);
  REQUIRE_THROWS_WITH(
    f.SetDOFHolderDataUnchecked(pcms::MakeConstRank2View(data)),
    ContainsSubstring("held by an external library"));
  REQUIRE_THROWS(source->CreateField<Real>("generic"));
}

TEST_CASE("ExternalSource: Evaluate rejects mismatched fields and tags")
{
  auto source = MockSource::Create();
  auto other = MockSource::Create();
  Points pts(source->GetCoordinateSystem(), {{0.5, 0.0, 0.0}});
  auto evaluator = source->CreatePointEvaluator<Real>(
    EvaluationRequest::FromCoordinates(pts.View()));

  auto foreign = other->MakeScalar();
  REQUIRE_THROWS_WITH(EvaluateToHost(*evaluator, foreign, ValueBasis{}, 1, 1),
                      ContainsSubstring("not a quantity of this source"));

  // A scalar tag on a vector field fails the write-tag check.
  auto v = source->MakeVector();
  REQUIRE_THROWS(EvaluateToHost(*evaluator, v, ValueBasis{}, 1, 3));
  // Scalar tags carry no width, so the output shape is checked separately.
  auto s2 = source->MakeTwoScalars();
  REQUIRE_THROWS_WITH(EvaluateToHost(*evaluator, s2, ValueBasis{}, 1, 1),
                      ContainsSubstring("expected [1][2]"));
}

TEST_CASE("PoloidalPlaneEmbedding: 2D queries evaluate a 3D source at theta0")
{
  auto source = MockSource::Create();
  const Real theta0 = 0.7;
  pcms::TransformedSource on_plane(
    source, std::make_shared<pcms::PoloidalPlaneEmbedding>(theta0));
  REQUIRE(pcms::SameCoordinateSystem(on_plane.GetCoordinateSystem(),
                                     pcms::csys::CylindricalRZ::Create()));

  const std::vector<std::vector<Real>> rz = {{0.5, -0.2}, {1.2, 0.4}};
  Points pts(pcms::csys::CylindricalRZ::Create(), rz);
  auto evaluator = on_plane.CreatePointEvaluator<Real>(
    EvaluationRequest::FromCoordinates(pts.View()));

  auto f = source->MakeScalar();
  auto s2 = source->MakeTwoScalars();
  const auto scalar = EvaluateToHost(*evaluator, f, ValueBasis{}, 2, 1);
  const auto multi = EvaluateToHost(*evaluator, s2, ValueBasis{}, 2, 2);
  for (size_t i = 0; i < rz.size(); ++i) {
    CAPTURE(i);
    const Real x[3] = {rz[i][0], theta0, rz[i][1]};
    REQUIRE_THAT(scalar(i, 0),
                 WithinAbs(MockValues(MockQuantity::Scalar, x)[0], kTol));
    const auto es = MockValues(MockQuantity::TwoScalars, x);
    REQUIRE_THAT(multi(i, 0), WithinAbs(es[0], kTol));
    REQUIRE_THAT(multi(i, 1), WithinAbs(es[1], kTol));
  }

  auto v = source->MakeVector();
  REQUIRE_THROWS_WITH(evaluator->OutputBasis(v.GetData().GetValueBasis()),
                      ContainsSubstring("maps points only"));
}

TEST_CASE("ExternalSource: interpolation onto a poloidal-plane mesh")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildUnitSquare(lib, 1);
  auto target = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, pcms::csys::CylindricalRZ::Create(), "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  auto source = MockSource::Create();
  const Real theta0 = -0.3;
  pcms::TransformedSource on_plane(
    source, std::make_shared<pcms::PoloidalPlaneEmbedding>(theta0));
  pcms::Interpolator<Real> interpolator(on_plane, *target);

  auto f = source->MakeScalar();
  auto g = target->CreateFunction<Real>("g");
  interpolator.Apply(f, g);

  const auto coords = pcms::test::CopyCoordinatesToHost(
    target->GetLayout()->GetDOFHolderCoordinates().GetValues());
  const auto result = g.GetDOFHolderDataHost().GetValues();
  for (size_t i = 0; i < coords.extent(0); ++i) {
    CAPTURE(i);
    const Real x[3] = {coords(i, 0), theta0, coords(i, 1)};
    REQUIRE_THAT(result(i, 0),
                 WithinAbs(MockValues(MockQuantity::Scalar, x)[0], kTol));
  }
}
