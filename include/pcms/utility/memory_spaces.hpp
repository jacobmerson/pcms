#ifndef PCMS_UTILITY_MEMORY_SPACES_HPP
#define PCMS_UTILITY_MEMORY_SPACES_HPP
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace pcms
{
// struct CudaMemorySpace{};
// struct HostMemorySpace{};
// using CudaMemorySpace = Kokkos::Cuda;
using HostMemorySpace = Kokkos::HostSpace;
using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;

// DeviceMemorySpace is the memory space of the default GPU execution space when
// Kokkos is built with a GPU backend. When no GPU backend is present it aliases
// HostMemorySpace so that generic code using DeviceMemorySpace compiles without
// change. PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE is defined only when the device
// memory space actually differs from the host memory space; use it to guard
// device-only overloads.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||               \
  defined(KOKKOS_ENABLE_SYCL)
#define PCMS_HAS_DISTINCT_DEVICE_MEMORY_SPACE
using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
#else
using DeviceMemorySpace = HostMemorySpace;
#endif

/// Kernels over at most this many items run as plain loops when the device
/// memory is host memory: opening a parallel region (about a microsecond with
/// OpenMP) costs more than such a loop, and dominates a bind of a few points.
inline constexpr std::size_t SmallHostBatchSize = 256;

/// @return whether a kernel over `n` items on DeviceMemorySpace runs as a
///         plain loop on the calling thread: the memory is host-accessible
///         and `n` is at most SmallHostBatchSize
[[nodiscard]] constexpr bool IsSmallHostBatch(std::size_t n) noexcept
{
  return Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                    DeviceMemorySpace>::accessible &&
         n <= SmallHostBatchSize;
}

} // namespace pcms

#endif // PCMS_UTILITY_MEMORY_SPACES_HPP
