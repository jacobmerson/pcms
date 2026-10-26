#include "pcms/field/bound_source.hpp"
#include "pcms/field/function_space.h"

namespace pcms
{

namespace detail
{

namespace
{

// A layout's DOF-holder coordinates live as long as the layout, which the
// request holds; raw query points belong to the caller and are copied.
Kokkos::View<Real**, DeviceMemorySpace> OwnedPoints(
  const EvaluationRequest& request)
{
  if (request.query_layout != nullptr) {
    return {};
  }
  const auto in = request.coords.GetValues();
  Kokkos::View<Real**, DeviceMemorySpace> points("deferred_bound_points",
                                                 in.extent(0), in.extent(1));
  const auto num_dims = static_cast<int>(in.extent(1));
  Kokkos::parallel_for(
    "deferred_bound_copy_points",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(in.extent(0))),
    KOKKOS_LAMBDA(const LO i) {
      for (int d = 0; d < num_dims; ++d) {
        points(i, d) = in(i, d);
      }
    });
  Kokkos::fence();
  return points;
}

EvaluationRequest Rebind(const EvaluationRequest& request,
                         const Kokkos::View<Real**, DeviceMemorySpace>& points)
{
  if (request.query_layout != nullptr) {
    return request;
  }
  return EvaluationRequest::FromCoordinates(
    CoordinateView<DeviceMemorySpace>(request.coords.GetCoordinateSystem(),
                                      MakeConstRank2View(points)),
    request.policy);
}

} // namespace

DeferredBoundSource::DeferredBoundSource(const EvaluationRequest& request)
  : BoundSource(request),
    points_(OwnedPoints(request)),
    request_(Rebind(request, points_))
{
}

} // namespace detail

} // namespace pcms
