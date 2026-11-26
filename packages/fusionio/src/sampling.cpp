#include "fusionio/sampling.hpp"
#include <pcms/field/query_points.hpp>
#include <pcms/field/value_view.hpp>
#include <pcms/utility/arrays.hpp>
#include <pcms/utility/assert.hpp>
#include <pcms/utility/coordinate_view.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace fusionio
{

namespace
{

using pcms::DeviceMemorySpace;
using pcms::Real;

constexpr int MaxDimension = 6;

// The box on the device.
struct Box
{
  int dim;
  Real lo[MaxDimension];
  Real hi[MaxDimension];
  bool sqrt[MaxDimension];
};

// splitmix64's finalizer
KOKKOS_INLINE_FUNCTION std::uint64_t Mix(std::uint64_t x)
{
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// The draw `counter` of the stream `seed`, uniform in [0, 1).
KOKKOS_INLINE_FUNCTION Real Uniform(std::uint64_t seed, std::uint64_t counter)
{
  constexpr std::uint64_t Golden = 0x9E3779B97F4A7C15ULL;
  const std::uint64_t bits = Mix(Mix(seed) + (counter + 1) * Golden) >> 11;
  return static_cast<Real>(bits) * (1.0 / 9007199254740992.0);
}

void Validate(const pcms::Function<Real>& f, const Domain& domain,
              const SamplingOptions& options)
{
  const auto dim = f.GetSpace().GetCoordinateSystem()->Dimension();
  if (f.GetLayout().GetNumComponents() != 1) {
    throw pcms::pcms_error("fusionio::SampleMarkers: the function is not a "
                           "scalar");
  }
  if (static_cast<int>(domain.size()) != dim || dim > MaxDimension) {
    throw pcms::pcms_error("fusionio::SampleMarkers: the domain has " +
                           std::to_string(domain.size()) +
                           " ranges, the function's coordinate system has " +
                           std::to_string(dim) + " coordinates (at most " +
                           std::to_string(MaxDimension) + ")");
  }
  for (const auto& range : domain) {
    if (!(range.lo <= range.hi)) {
      throw pcms::pcms_error("fusionio::SampleMarkers: a range has lo > hi");
    }
  }
  if (options.batch <= 0) {
    throw pcms::pcms_error("fusionio::SampleMarkers: batch must be positive");
  }
}

// The rounds. `rows` holds the markers of a round from its first row on when
// `sink` is given (and has at least `batch` rows), else all the markers, at
// the number accepted so far. Returns the candidates drawn.
std::size_t Run(const pcms::Function<Real>& f, const Domain& domain,
                const SamplingOptions& options, const MarkerRows& rows,
                const MarkerSink* sink)
{
  Box box{};
  box.dim = static_cast<int>(domain.size());
  for (int d = 0; d < box.dim; ++d) {
    box.lo[d] = domain[d].lo;
    box.hi[d] = domain[d].hi;
    box.sqrt[d] = domain[d].draw == Draw::Sqrt;
  }
  const int dim = box.dim;
  const int batch = options.batch;
  const std::size_t samples = options.samples;
  const std::uint64_t seed = options.seed;
  const Real threshold = options.threshold;

  // allocated once: the candidates, their values, their slots
  Kokkos::View<Real**, DeviceMemorySpace> candidates(
    Kokkos::view_alloc("fusionio candidates", Kokkos::WithoutInitializing),
    batch, dim);
  Kokkos::View<Real**, DeviceMemorySpace> values(
    Kokkos::view_alloc("fusionio values", Kokkos::WithoutInitializing), batch,
    1);
  Kokkos::View<int*, DeviceMemorySpace> slots(
    Kokkos::view_alloc("fusionio slots", Kokkos::WithoutInitializing), batch);

  const auto geometry = f.GetSpace().GetGeometry();
  pcms::QueryPoints query(
    geometry, pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL,
                                      std::numeric_limits<Real>::quiet_NaN()});
  const auto basis = f.GetData().GetValueBasis();

  std::size_t accepted = 0;
  std::size_t drawn = 0;
  while (accepted < samples) {
    const std::uint64_t first = drawn;
    Kokkos::parallel_for(
      "fusionio draw candidates",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, batch),
      KOKKOS_LAMBDA(const int i) {
        for (int d = 0; d < box.dim; ++d) {
          Real t = Uniform(seed, (first + i) * MaxDimension + d);
          if (box.sqrt[d]) {
            t = Kokkos::sqrt(t);
          }
          candidates(i, d) = box.lo[d] + t * (box.hi[d] - box.lo[d]);
        }
      });
    drawn += batch;

    query.Locate(pcms::CoordinateView<DeviceMemorySpace>(
      geometry->GetCoordinateSystem(), pcms::MakeConstRank2View(candidates)));
    query.Evaluate(f, pcms::ValueView<Real, DeviceMemorySpace>(
                        basis, pcms::MakeRank2View(values)));

    // slot of each accepted candidate (-1: rejected): an exclusive scan of the
    // acceptance, so the order does not depend on the threads
    int num_accepted = 0;
    Kokkos::parallel_scan(
      "fusionio accept",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, batch),
      KOKKOS_LAMBDA(const int i, int& sum, const bool final) {
        const int accept = values(i, 0) >= threshold ? 1 : 0;
        if (final) {
          slots(i) = accept ? sum : -1;
        }
        sum += accept;
      },
      num_accepted);

    const std::size_t take = std::min<std::size_t>(
      static_cast<std::size_t>(num_accepted), samples - accepted);
    const std::size_t offset = sink ? 0 : accepted;
    Kokkos::parallel_for(
      "fusionio write markers",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, batch),
      KOKKOS_LAMBDA(const int i) {
        const int slot = slots(i);
        if (slot >= 0 && static_cast<std::size_t>(slot) < take) {
          for (int d = 0; d < box.dim; ++d) {
            rows(offset + slot, d) = candidates(i, d);
          }
          rows(offset + slot, box.dim) = values(i, 0);
        }
      });
    accepted += take;
    if (sink && take > 0) {
      (*sink)(Kokkos::subview(
        rows, std::pair<std::size_t, std::size_t>(0, take), Kokkos::ALL));
    }
  }
  return drawn;
}

} // namespace

Markers SampleMarkers(const pcms::Function<Real>& f, const Domain& domain,
                      const SamplingOptions& options)
{
  Validate(f, domain, options);
  Markers markers;
  markers.rows = MarkerRows(
    Kokkos::view_alloc("fusionio markers", Kokkos::WithoutInitializing),
    options.samples, domain.size() + 1);
  markers.candidates = Run(f, domain, options, markers.rows, nullptr);
  return markers;
}

std::size_t SampleMarkers(const pcms::Function<Real>& f, const Domain& domain,
                          const SamplingOptions& options,
                          const MarkerSink& sink)
{
  Validate(f, domain, options);
  const MarkerRows rows(
    Kokkos::view_alloc("fusionio chunk", Kokkos::WithoutInitializing),
    static_cast<std::size_t>(options.batch), domain.size() + 1);
  return Run(f, domain, options, rows, &sink);
}

} // namespace fusionio
