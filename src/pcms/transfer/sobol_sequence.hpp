#ifndef PCMS_TRANSFER_SOBOL_SEQUENCE_HPP
#define PCMS_TRANSFER_SOBOL_SEQUENCE_HPP

#include "pcms/utility/memory_spaces.h"
#include <Kokkos_Core.hpp>
#include <cstdint>

namespace pcms
{

// Computes the index-th point of the unscrambled 2D Sobol sequence using the
// Gray-code construction. Dimension 0 reduces to the van der Corput sequence
// in base 2; dimension 1 uses the degree-1 primitive polynomial x + 1, whose
// Joe-Kuo direction integers follow m_k = m_{k-1} XOR (m_{k-1} << 1)
// (1, 3, 5, 15, 17, ...). Index 0 yields the origin; callers that want to
// avoid (0, 0) should start at index 1.
//
// Each point depends only on its index, so this is safe to call from device
// kernels with one thread per point.
KOKKOS_INLINE_FUNCTION Kokkos::Array<double, 2> Sobol2DPoint(uint32_t index)
{
  const uint32_t gray = index ^ (index >> 1);
  uint32_t x_bits = 0;
  uint32_t y_bits = 0;
  uint32_t m = 1;
  for (int k = 0; k < 32; ++k) {
    if (gray & (1u << k)) {
      x_bits ^= 1u << (31 - k);
      y_bits ^= m << (31 - k);
    }
    m ^= m << 1;
  }
  constexpr double scale = 1.0 / 4294967296.0; // 2^-32
  return {x_bits * scale, y_bits * scale};
}

// Returns n consecutive 2D Sobol points starting at start_index (default 1,
// skipping the origin) as a device view of shape [n][2], filled in parallel.
[[nodiscard]] inline Kokkos::View<double**, DeviceMemorySpace>
GetSobolSequence(int n, uint32_t start_index = 1)
{
  Kokkos::View<double**, DeviceMemorySpace> points("sobol_points", n, 2);
  Kokkos::parallel_for(
    "fill_sobol_sequence",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
      const auto p = Sobol2DPoint(start_index + static_cast<uint32_t>(i));
      points(i, 0) = p[0];
      points(i, 1) = p[1];
    });
  return points;
}

} // namespace pcms

#endif // PCMS_TRANSFER_SOBOL_SEQUENCE_HPP
