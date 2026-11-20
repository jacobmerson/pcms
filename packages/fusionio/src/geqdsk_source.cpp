#include "fusionio/geqdsk.hpp"
#include "geqdsk_reader.hpp"
#include "host_evaluation.hpp"
#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <set>

namespace fusionio
{

namespace
{

using namespace pcms;

constexpr Real TwoPi = 2 * std::numbers::pi;

using Rows = Kokkos::View<Real**, DeviceMemorySpace>;
using FoundIndices = Kokkos::View<const LO*, DeviceMemorySpace>;

// The kernels are free functions because nvcc does not take an extended
// lambda in a private member function.

// The (R, Z) of every (R, phi, Z) point.
template <typename Points, typename RZ>
void ProjectToPlane(LO n, const Points& points, const RZ& rz)
{
  Kokkos::parallel_for(
    "GeqdskBinding::Project",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const LO i) {
      rz(i, 0) = points(i, 0);
      rz(i, 1) = points(i, 2);
    });
}

// The found mask of n points of which `missing` are not found.
template <typename Missing>
Kokkos::View<bool*, DeviceMemorySpace> FoundMask(LO n, const Missing& missing)
{
  Kokkos::View<bool*, DeviceMemorySpace> found("geqdsk_found", n);
  Kokkos::deep_copy(found, true);
  Kokkos::parallel_for(
    "GeqdskBinding::Missing",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(missing.extent(0))),
    KOKKOS_LAMBDA(const LO k) { found(missing(k)) = false; });
  return found;
}

// Points of a geqdsk source: the (R, Z) projection of the points bound on the
// (R, Z) grid's geometry. The found points are the projection's.
class GeqdskBinding final : public BoundPoints
{
public:
  GeqdskBinding(std::shared_ptr<const Geometry> geometry,
                const BindRequest& request,
                std::shared_ptr<const Geometry> plane)
    : BoundPoints(std::move(geometry), request), plane_(std::move(plane))
  {
    Locate();
  }

  /// The projection's binding: its points are the (R, Z) of every point.
  [[nodiscard]] const std::shared_ptr<const BoundPoints>& Plane() const noexcept
  {
    return plane_bound_;
  }

protected:
  void RebindImpl() override { Locate(); }

  [[nodiscard]] bool UnsharedImpl() const noexcept override
  {
    return rz_.use_count() <= 1 && plane_bound_.use_count() <= 1;
  }

private:
  void Locate()
  {
    const LO n = NumPoints();
    if (rz_.extent(0) < static_cast<size_t>(n)) {
      rz_ = Rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, "geqdsk_rz"),
                 n, 2);
    }
    const auto rz =
      Kokkos::subview(rz_, Kokkos::make_pair(LO{0}, n), Kokkos::ALL);
    ProjectToPlane(n, LocatedPointValues(), rz);
    const CoordinateView<DeviceMemorySpace> projected(
      plane_->GetCoordinateSystem(), MakeRank2View(rz));
    plane_bound_ = plane_bound_ == nullptr
                     ? plane_->Bind(projected, Policy())
                     : plane_->Rebind(std::move(plane_bound_), projected);
    const auto missing = plane_bound_->MissingPoints();
    if (missing.extent(0) > 0) {
      SetFound(FoundMask(n, missing));
    }
  }

  std::shared_ptr<const Geometry> plane_;
  std::shared_ptr<const BoundPoints> plane_bound_;
  Rows rz_;
};

// (R, phi, Z) points on the file's (R, Z) grid, CylindricalRThetaZ as
// Fusion-IO's (R, phi, Z): psi does not depend on phi.
class GeqdskGeometry final : public Geometry
{
public:
  explicit GeqdskGeometry(std::shared_ptr<const Geometry> plane)
    : plane_(std::move(plane)), system_(csys::CreateCylindricalRThetaZ())
  {
  }

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return system_;
  }

protected:
  [[nodiscard]] std::shared_ptr<BoundPoints> BindImpl(
    const BindRequest& request) const override
  {
    return std::make_shared<GeqdskBinding>(shared_from_this(), request, plane_);
  }

private:
  std::shared_ptr<const Geometry> plane_;
  std::shared_ptr<const CoordinateSystem> system_;
};

// What the quantities are made of, shared by the source and its evaluators.
struct GeqdskFunctions
{
  // psi on the (R, Z) grid
  Function<Real> psi;
  // F and p of psi
  Function<Real> current, pressure;
  // psi at the axis and at the boundary, and the profiles' grid, which
  // is [psi_min, psi_max] as the grid computes its end
  Real psi_axis, psi_boundary, psi_min, psi_max;
};

// The values of `function` at the points of `evaluator`, one row each.
Rows EvaluateRows(const PointEvaluator<Real>& evaluator,
                  const Function<Real>& function, int columns)
{
  Rows out("geqdsk_values", evaluator.NumPoints(), columns);
  evaluator.Evaluate(
    function, ValueView<Real, DeviceMemorySpace>(
                evaluator.OutputBasis(function.GetData().GetValueBasis()),
                MakeRank2View(out)));
  return out;
}

// psi clamped to [lo, hi] in place; a NaN becomes lo. The profiles are
// constant beyond their ends.
void ClampRows(const Rows& psi, Real lo, Real hi)
{
  Kokkos::parallel_for(
    "GeqdskEvaluator::Clamp",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(psi.extent(0))),
    KOKKOS_LAMBDA(const LO i) {
      const Real v = psi(i, 0);
      psi(i, 0) = !(v >= lo) ? lo : (v > hi ? hi : v);
    });
}

// out = (psi - offset) * factor at the found points
template <typename Out>
void ScaleValues(const Out& out, const Rows& psi, const FoundIndices& found,
                 Real offset, Real factor)
{
  Kokkos::parallel_for(
    "GeqdskEvaluator::Scale",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(found.extent(0))),
    KOKKOS_LAMBDA(const LO k) {
      const LO i = found(k);
      out(i, 0) = (psi(i, 0) - offset) * factor;
    });
}

// The gradient of a scalar a * psi + b: (d/dR, (1/R) d/dphi, d/dZ)
template <typename Out>
void ScaleGradients(const Out& out, const Rows& grad, const FoundIndices& found,
                    Real factor)
{
  Kokkos::parallel_for(
    "GeqdskEvaluator::ScaleGradient",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(found.extent(0))),
    KOKKOS_LAMBDA(const LO k) {
      const LO i = found(k);
      out(i, 0) = grad(i, 0) * factor;
      out(i, 1) = 0;
      out(i, 2) = grad(i, 1) * factor;
    });
}

// out = the first column of `values` at the found points
template <typename Out>
void CopyValues(const Out& out, const Rows& values, const FoundIndices& found)
{
  Kokkos::parallel_for(
    "GeqdskEvaluator::Copy",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(found.extent(0))),
    KOKKOS_LAMBDA(const LO k) {
      const LO i = found(k);
      out(i, 0) = values(i, 0);
    });
}

// B = F grad(phi) + grad(phi) x grad(psi) at the found points, in (R, phi, Z)
template <typename Out, typename RZ>
void MagneticFieldRows(const Out& out, const RZ& rz, const Rows& grad,
                       const Rows& current, const FoundIndices& found)
{
  Kokkos::parallel_for(
    "GeqdskEvaluator::MagneticField",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(found.extent(0))),
    KOKKOS_LAMBDA(const LO k) {
      const LO i = found(k);
      const Real r = rz(i, 0);
      out(i, 0) = grad(i, 1) / r;
      out(i, 1) = current(i, 0) / r;
      out(i, 2) = -grad(i, 0) / r;
    });
}

using GeqdskFieldData = detail::ExternalFieldData<Quantity>;

// Evaluates the quantities at the points of a GeqdskGeometry binding: the
// value, or for the flux its gradient (d/dR, (1/R) d/dphi, d/dZ), physical
// components. psi and its gradient come from the PCMS spline on the (R, Z)
// points the binding projected to; the profiles from the PCMS spline in psi,
// bound at the psi found (one binding and one evaluation per Evaluate).
class GeqdskEvaluator final : public PointEvaluator<Real>
{
public:
  GeqdskEvaluator(std::shared_ptr<const GeqdskBinding> bound,
                  DifferentialOperator op,
                  std::shared_ptr<const FieldLayout> layout,
                  std::shared_ptr<const GeqdskFunctions> functions)
    : PointEvaluator<Real>(bound, op, std::move(layout)),
      bound_(std::move(bound)),
      functions_(std::move(functions))
  {
    const auto& space = functions_->psi.GetSpace();
    value_ = space.CreatePointEvaluator<Real>(bound_->Plane(),
                                              DifferentialOperator::Value);
    gradient_ = space.CreatePointEvaluator<Real>(
      bound_->Plane(), DifferentialOperator::Gradient);
  }

protected:
  void EvaluateImpl(const Field<Real>& field,
                    ValueView<Real, DeviceMemorySpace> values) const override
  {
    const auto* data = dynamic_cast<const GeqdskFieldData*>(&field.GetData());
    if (data == nullptr) {
      throw pcms_error("GeqdskEvaluator: the field is not a geqdsk quantity");
    }
    const auto& f = *functions_;
    const Quantity quantity = data->quantity;
    const auto out = values.GetValues();
    const auto found = bound_->FoundPoints();
    // psi's affine map of the flux quantities: (psi - offset) * factor
    const Real offset = quantity == Quantity::PoloidalFluxNorm ? f.psi_axis : 0;
    const Real factor = quantity == Quantity::PoloidalFlux
                          ? TwoPi
                          : 1 / (f.psi_boundary - f.psi_axis);

    if (Operator() == DifferentialOperator::Gradient) {
      ScaleGradients(out, EvaluateRows(*gradient_, f.psi, 2), found, factor);
    } else if (quantity == Quantity::MagneticField) {
      const auto grad = EvaluateRows(*gradient_, f.psi, 2);
      const auto current = EvaluateProfile(f.current);
      MagneticFieldRows(out, bound_->Plane()->LocatedPointValues(), grad,
                        current, found);
    } else if (quantity == Quantity::TotalPressure) {
      CopyValues(out, EvaluateProfile(f.pressure), found);
    } else {
      ScaleValues(out, EvaluateRows(*value_, f.psi, 1), found, offset, factor);
    }
  }

private:
  // `profile` at psi at the points, which is clamped to the profile's grid
  Rows EvaluateProfile(const Function<Real>& profile) const
  {
    const auto& f = *functions_;
    auto psi = EvaluateRows(*value_, f.psi, 1);
    ClampRows(psi, f.psi_min, f.psi_max);
    const auto& space = profile.GetSpace();
    const CoordinateView<DeviceMemorySpace> coordinates(
      space.GetCoordinateSystem(), MakeRank2View(psi));
    const auto evaluator = space.CreatePointEvaluator<Real>(
      space.GetGeometry()->Bind(coordinates, OutOfBoundsPolicy{}),
      DifferentialOperator::Value);
    return EvaluateRows(*evaluator, profile, 1);
  }

  std::shared_ptr<const GeqdskBinding> bound_;
  std::shared_ptr<const GeqdskFunctions> functions_;
  std::unique_ptr<PointEvaluator<Real>> value_, gradient_;
};

// The values of a spline's grid in a function of its space.
Function<Real> CreateSplineFunction(const FunctionSpace& space,
                                    const std::string& name,
                                    const std::vector<Real>& values)
{
  auto function = space.CreateFunction<Real>(name);
  function.SetDOFHolderDataHost(ValueView<const Real, HostMemorySpace>(
    function.GetData().GetValueBasis(),
    Rank2View<const Real, HostMemorySpace>(values.data(), values.size(), 1)));
  return function;
}

class GeqdskSourceImpl final : public GeqdskSource
{
public:
  explicit GeqdskSourceImpl(const std::string& path)
  {
    const detail::GeqdskData data = detail::ReadGeqdsk(path);
    boundary_ = data.boundary;
    limiter_ = data.limiter;
    axis_r_ = data.rmaxis;
    axis_z_ = data.zmaxis;

    UniformGrid2D grid;
    grid.divisions = {data.nw - 1, data.nh - 1};
    grid.edge_length = {data.rdim, data.zdim};
    grid.bot_left = {data.rleft, data.zmid - data.zdim / 2};
    box_ = {{data.rleft, 0, grid.bot_left[1]},
            {data.rleft + data.rdim, TwoPi, grid.bot_left[1] + data.zdim}};
    const auto plane = CreateSplineSpace(
      uniform_grid::CreateGeometry(grid, csys::CreateCylindricalRZ()));

    // The profiles' grid runs from the axis to the boundary; the spline
    // space's grid increases, so a profile of a flux that decreases outward
    // is stored reversed.
    const bool reversed = data.sibry < data.simag;
    UniformGrid1D profile_grid;
    profile_grid.divisions = {data.nw - 1};
    profile_grid.edge_length = {std::abs(data.sibry - data.simag)};
    profile_grid.bot_left = {std::min(data.simag, data.sibry)};
    profile_space_ = CreateSplineSpace(
      uniform_grid::CreateGeometry(profile_grid, csys::CreateCartesian(1)));
    for (const auto* values : {&data.fpol, &data.pres, &data.qpsi}) {
      profiles_.push_back(*values);
      if (reversed) {
        std::reverse(profiles_.back().begin(), profiles_.back().end());
      }
    }
    functions_ = std::make_shared<GeqdskFunctions>(
      GeqdskFunctions{CreateSplineFunction(*plane, "psi", data.psirz),
                      CreateProfile(GeqdskProfile::PoloidalCurrent),
                      CreateProfile(GeqdskProfile::Pressure), data.simag,
                      data.sibry, profile_grid.bot_left[0],
                      profile_grid.bot_left[0] + profile_grid.edge_length[0]});

    geometry_ = std::make_shared<GeqdskGeometry>(plane->GetGeometry());
    const auto create = [this](int components,
                               std::set<DifferentialOperator> supported) {
      return detail::CreateExternalSpace<GeqdskBinding>(
        "GeqdskGeometry", geometry_, components, std::move(supported),
        [functions = functions_](std::shared_ptr<const GeqdskBinding> bound,
                                 DifferentialOperator op,
                                 std::shared_ptr<const FieldLayout> layout) {
          return std::make_unique<GeqdskEvaluator>(
            std::move(bound), op, std::move(layout), functions);
        });
    };
    fluxes_ = create(1, {DifferentialOperator::Gradient});
    scalars_ = create(1, {});
    vectors_ = create(3, {});
  }

  [[nodiscard]] std::shared_ptr<const Geometry> GetGeometry() const override
  {
    return geometry_;
  }

  [[nodiscard]] std::vector<QuantityInfo> Quantities() const override
  {
    return {{Quantity::TotalPressure, 1},
            {Quantity::PoloidalFlux, 1},
            {Quantity::PoloidalFluxNorm, 1},
            {Quantity::MagneticField, 3}};
  }

  [[nodiscard]] std::vector<Real> Times() const override { return {}; }

  [[nodiscard]] Real ToroidalPeriod() const override { return TwoPi; }

  [[nodiscard]] std::optional<Box> BoundingBox() const override { return box_; }

  [[nodiscard]] std::optional<Real> EvaluateSeries(Series series,
                                                   Real /*time*/) const override
  {
    switch (series) {
      case Series::MagneticAxisPsi: return functions_->psi_axis;
      case Series::LcfsPsi: return functions_->psi_boundary;
      case Series::MagneticAxisR: return axis_r_;
      case Series::MagneticAxisZ: return axis_z_;
    }
    return std::nullopt;
  }

  [[nodiscard]] Function<Real> CreateFunction(
    Quantity quantity, const Options& options) const override
  {
    if (options.timeslice || options.part || options.species ||
        options.linear_scale || options.phase) {
      throw pcms_error("fusionio: a geqdsk source has no options");
    }
    const auto space = [&]() -> const std::shared_ptr<const FunctionSpace>& {
      switch (quantity) {
        case Quantity::PoloidalFlux:
        case Quantity::PoloidalFluxNorm: return fluxes_;
        case Quantity::TotalPressure: return scalars_;
        case Quantity::MagneticField: return vectors_;
        default: break;
      }
      std::vector<std::string> available;
      for (const auto& info : Quantities()) {
        available.emplace_back(Name(info.quantity));
      }
      detail::ThrowUnknownQuantity("fusionio", Name(quantity), available);
    }();
    return space->CreateFunction<Real>(
      std::string(Name(quantity)),
      std::make_unique<GeqdskFieldData>(
        quantity, detail::QuantityBasis(quantity == Quantity::MagneticField)));
  }

  [[nodiscard]] const Polyline& Boundary() const override { return boundary_; }

  [[nodiscard]] const Polyline& Limiter() const override { return limiter_; }

  [[nodiscard]] Function<Real> CreateProfile(
    GeqdskProfile profile) const override
  {
    static constexpr const char* Names[] = {"F", "p", "q"};
    const auto index = static_cast<size_t>(profile);
    return CreateSplineFunction(*profile_space_, Names[index],
                                profiles_.at(index));
  }

private:
  Polyline boundary_, limiter_;
  Real axis_r_ = 0, axis_z_ = 0;
  Box box_{};
  // the grid of each profile, in the order of GeqdskProfile
  std::shared_ptr<const FunctionSpace> profile_space_;
  std::vector<std::vector<Real>> profiles_;
  std::shared_ptr<const GeqdskFunctions> functions_;
  std::shared_ptr<const GeqdskGeometry> geometry_;
  std::shared_ptr<const FunctionSpace> fluxes_, scalars_, vectors_;
};

} // namespace

std::shared_ptr<GeqdskSource> OpenGeqdsk(const std::string& path)
{
  return std::make_shared<GeqdskSourceImpl>(path);
}

} // namespace fusionio
