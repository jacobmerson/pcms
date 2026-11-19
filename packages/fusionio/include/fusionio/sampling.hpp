#ifndef FUSIONIO_SAMPLING_HPP
#define FUSIONIO_SAMPLING_HPP

/// Rejection sampling of markers from a function: points drawn in a box of the
/// function's coordinates, kept where the function's value reaches a
/// threshold. The function is evaluated in batches on the device and memory
/// is bounded by one batch of candidates and the markers.

#include <pcms/field/field.hpp>
#include <pcms/utility/memory_spaces.hpp>
#include <pcms/utility/types.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace fusionio
{

/// How the candidates of an axis are drawn.
enum class Draw
{
  /// uniformly in [lo, hi]
  Uniform,
  /// lo + sqrt(u) (hi - lo) for u uniform in [0, 1): uniform in area for a
  /// radial axis from the centre
  Sqrt
};

/// The range of one axis of the box candidates are drawn in.
struct Range
{
  pcms::Real lo;
  pcms::Real hi;
  Draw draw = Draw::Uniform;
};

/// The box candidates are drawn in: one Range per coordinate of the function's
/// coordinate system (at most 6).
using Domain = std::vector<Range>;

struct SamplingOptions
{
  /// markers to produce
  std::size_t samples = 0;
  /// candidates per round, at most INT_MAX; they and their values are the
  /// memory of a round
  int batch = 1000000;
  /// the candidates are a function of (seed, index of the candidate), so a
  /// run gives the same markers on any number of threads or devices
  std::uint64_t seed = 14000;
  /// a candidate is kept if the function's value is >= threshold (a value
  /// the function does not have, e.g. outside its domain, is not)
  pcms::Real threshold = 0;
};

/// Markers on the device: row i is the coordinates of marker i followed by
/// the function's value there.
using MarkerRows =
  Kokkos::View<pcms::Real**, Kokkos::LayoutRight, pcms::DeviceMemorySpace>;

struct Markers
{
  /// samples x (dimension + 1)
  MarkerRows rows;
  /// candidates drawn: rows.extent(0) / candidates is the acceptance
  std::size_t candidates = 0;
};

/// Receives the markers of a round (rows of up to `batch` markers, valid
/// during the call).
using MarkerSink = std::function<void(const MarkerRows& rows)>;

/// Draws candidates until `options.samples` of them are kept, in rounds of
/// `options.batch`: the candidates of a round are drawn in place, located on
/// the function's geometry (one pcms::QueryPoints, rebound each round),
/// evaluated, and the accepted ones are written to the markers at the slots
/// of an exclusive scan of the acceptance, so the markers are in the order of
/// the candidates and the last round's are clipped at `samples`. The function
/// must be a scalar with data on a geometry whose coordinate system has
/// domain.size() coordinates. A threshold nothing reaches never returns.
/// @throws pcms::pcms_error if the function, domain or options are invalid
[[nodiscard]] Markers SampleMarkers(const pcms::Function<pcms::Real>& f,
                                    const Domain& domain,
                                    const SamplingOptions& options);

/// As above, delivering each round's accepted markers to `sink` (in order)
/// instead of keeping them, so memory does not grow with `samples`.
/// @return the candidates drawn
std::size_t SampleMarkers(const pcms::Function<pcms::Real>& f,
                          const Domain& domain, const SamplingOptions& options,
                          const MarkerSink& sink);

} // namespace fusionio

#endif // FUSIONIO_SAMPLING_HPP
