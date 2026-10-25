#ifndef PCMS_FIELD_EVALUATOR_HOST_POINT_EVALUATOR_HPP
#define PCMS_FIELD_EVALUATOR_HOST_POINT_EVALUATOR_HPP

#include "pcms/field/evaluation_request.h"
#include "pcms/field/field.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/value_view.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <string>
#include <utility>
#include <vector>

namespace pcms
{

/// Point evaluator that drives a host-side backend one point at a time, for
/// sources evaluated by an external library (e.g. NIMROD, Fusion-IO) that
/// can neither run on the device nor evaluate many points per call.
///
/// The query points are copied to the host once, at construction. Each point
/// is then located with the backend, in order, which fixes the out-of-bounds
/// set: ERROR throws if any point is outside the source's domain, FILL records
/// those points in FilledPoints() and writes the fill value to their rows, and
/// NEAREST_BOUNDARY is rejected. Evaluate only asks the backend for located
/// points, so any localization state the backend caches per point while
/// locating (search hints, logical-coordinate guesses) is reused by every
/// Evaluate call.
///
/// Backend must provide:
///   bool Locate(LO i, const Real* x);
///     Called once per query point, in order, at construction. x holds the
///     point's coordinates (the request's coordinate-system dimension).
///     Returns false when the point is outside the source's domain.
///   Handle Resolve(const Field<Real>& field);
///     Called once per Evaluate. Throws pcms_error unless `field` was created
///     by this backend's source; returns whatever EvaluatePoint needs to
///     identify the field.
///   bool EvaluatePoint(const Handle& handle, LO i, const Real* x, Real* out);
///     Writes the field's components at located point i into out. Returns
///     false if the backend cannot evaluate a point Locate accepted.
///
/// The backend's methods are called sequentially from Evaluate, which is
/// const; external evaluation libraries are generally neither thread safe nor
/// reentrant, so Evaluate must not be called concurrently on one evaluator.
template <typename Backend>
class HostPointEvaluator final : public PointEvaluator<Real>
{
public:
  /// @param backend per-point evaluation backend; owned by the evaluator
  /// @param request query points and out-of-bounds policy
  /// @param name label used in error messages (e.g. the source's type)
  HostPointEvaluator(Backend backend, const EvaluationRequest& request,
                     std::string name)
    : backend_(std::move(backend)),
      name_(std::move(name)),
      fill_value_(request.policy.fill_value)
  {
    PCMS_FUNCTION_TIMER;
    if (request.policy.mode == OutOfBoundsMode::NEAREST_BOUNDARY) {
      throw pcms_error(name_ +
                       ": nearest-boundary evaluation is not supported");
    }
    CopyPointsToHost(request.coords.GetValues());
    Localize(request.policy.mode);
  }

  void Evaluate(const Field<Real>& field,
                ValueView<Real, DeviceMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    this->CheckEvaluateWriteTag(name_.c_str(), field, values);
    const auto out = values.GetValues();
    const auto num_components =
      static_cast<size_t>(field.GetLayout().GetNumComponents());
    if (out.extent(0) != static_cast<size_t>(NumPoints()) ||
        out.extent(1) != num_components) {
      throw pcms_error(name_ + "::Evaluate: output is shaped [" +
                       std::to_string(out.extent(0)) + "][" +
                       std::to_string(out.extent(1)) + "] but expected [" +
                       std::to_string(NumPoints()) + "][" +
                       std::to_string(num_components) + "]");
    }

    const auto handle = backend_.Resolve(field);
    Kokkos::View<Real**, HostMemorySpace> host_out(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_point_eval_out"),
      out.extent(0), out.extent(1));
    for (LO i = 0; i < NumPoints(); ++i) {
      Real* row = &host_out(i, 0);
      if (!located_[i]) {
        for (size_t c = 0; c < num_components; ++c) {
          row[c] = fill_value_;
        }
        continue;
      }
      if (!backend_.EvaluatePoint(handle, i, &points_(i, 0), row)) {
        throw pcms_error(
          name_ + "::Evaluate: evaluation failed at query point " +
          std::to_string(i) + ", which was located inside the source's domain");
      }
    }

    auto device_out =
      Kokkos::create_mirror_view_and_copy(DeviceMemorySpace(), host_out);
    CopyDeviceRank2ViewToRank2View(out, MakeConstRank2View(device_out));
  }

  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> FilledPoints()
    const override
  {
    return filled_points_;
  }

  [[nodiscard]] LO NumPoints() const noexcept
  {
    return static_cast<LO>(points_.extent(0));
  }

private:
  void CopyPointsToHost(Rank2View<const Real, DeviceMemorySpace> coords)
  {
    // Copy element-wise on the device so any layout the caller's coordinates
    // carry lands in a LayoutRight host buffer whose rows are contiguous.
    Kokkos::View<Real**, Kokkos::LayoutRight, DeviceMemorySpace> staged(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_point_eval_coords"),
      coords.extent(0), coords.extent(1));
    Kokkos::parallel_for(
      "host_point_eval_stage_coords",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
        0, static_cast<LO>(coords.extent(0))),
      KOKKOS_LAMBDA(const LO i) {
        for (size_t d = 0; d < coords.extent(1); ++d) {
          staged(i, d) = coords(i, d);
        }
      });
    points_ = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), staged);
  }

  void Localize(OutOfBoundsMode mode)
  {
    located_.assign(static_cast<size_t>(NumPoints()), false);
    std::vector<LO> outside;
    for (LO i = 0; i < NumPoints(); ++i) {
      located_[i] = backend_.Locate(i, &points_(i, 0));
      if (!located_[i]) {
        outside.push_back(i);
      }
    }
    if (outside.empty()) {
      return;
    }
    if (mode == OutOfBoundsMode::ERROR) {
      throw pcms_error(name_ + ": " + std::to_string(outside.size()) +
                       " query point(s) are outside the source's domain "
                       "(first is point " +
                       std::to_string(outside.front()) + ")");
    }
    // Copy into an owning view: a mirror of the unmanaged host view would
    // alias `outside` when the device memory space is host memory.
    Kokkos::View<LO*, DeviceMemorySpace> filled(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "host_point_eval_filled"),
      outside.size());
    Kokkos::deep_copy(
      filled, Kokkos::View<const LO*, HostMemorySpace, Kokkos::MemoryUnmanaged>(
                outside.data(), outside.size()));
    filled_points_ = filled;
  }

  mutable Backend backend_;
  std::string name_;
  Real fill_value_;
  Kokkos::View<Real**, Kokkos::LayoutRight, HostMemorySpace> points_;
  std::vector<bool> located_;
  Kokkos::View<const LO*, DeviceMemorySpace> filled_points_;
};

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_HOST_POINT_EVALUATOR_HPP
