#ifndef PCMS_COUPLER_OVERLAP_MASK_HPP
#define PCMS_COUPLER_OVERLAP_MASK_HPP

#include "pcms/field/field_layout.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/detail/arrays.hpp"
#include "pcms/utility/assert.hpp"
#include <functional>

namespace pcms
{

/// @brief Whether each owned DOF holder, in owned order, is in the overlap
struct OverlapMask
{
  /// @brief All DOF holders in the overlap
  OverlapMask(size_t size) : is_overlap_("overlap_info", size)
  {
    Kokkos::deep_copy(is_overlap_, true);
  }

  /// @brief Overlap from a DOF holder's classification (dimension, id)
  /// @param in_overlap nonzero for a classification in the overlap; holders
  ///        without classification are passed as (-1, -1)
  explicit OverlapMask(std::function<int8_t(int, int)> in_overlap)
    : in_overlap_func_(std::move(in_overlap))
  {
  }

  /// @brief A precomputed mask, one entry per owned DOF holder
  OverlapMask(size_t size,
              Kokkos::View<const bool*, DeviceMemorySpace> is_overlap)
    : is_overlap_("overlap_info", size)
  {
    PCMS_ALWAYS_ASSERT(is_overlap.extent(0) == size);
    Kokkos::deep_copy(is_overlap_, is_overlap);
  }

  /// @brief The mask on the host, one entry per owned DOF holder of `layout`
  Kokkos::View<const bool*, HostMemorySpace> GetMask(
    const FieldLayout& layout) const
  {
    if (!in_overlap_func_) {
      PCMS_ALWAYS_ASSERT(is_overlap_.extent(0) ==
                         static_cast<size_t>(layout.GetNumDofHolder()));
      return Kokkos::create_mirror_view_and_copy(HostMemorySpace(),
                                                 is_overlap_);
    }
    const auto class_dims_device =
      layout.GetDOFHolderClassificationDimensions();
    const auto class_ids_device = layout.GetDOFHolderClassificationIds();
    const size_t n = class_dims_device.size();
    Kokkos::View<LO*, HostMemorySpace> class_dims("class_dims", n);
    Kokkos::View<LO*, HostMemorySpace> class_ids("class_ids", n);
    detail::CopyDeviceRank1ViewToHostView<LO>(class_dims, class_dims_device);
    detail::CopyDeviceRank1ViewToHostView<LO>(class_ids, class_ids_device);
    Kokkos::View<bool*, HostMemorySpace> is_overlap("overlap_info", n);
    for (size_t i = 0; i < n; ++i) {
      is_overlap[i] =
        static_cast<bool>(in_overlap_func_(class_dims[i], class_ids[i]));
    }
    return is_overlap;
  }

private:
  Kokkos::View<bool*, DeviceMemorySpace> is_overlap_;
  std::function<int8_t(int, int)> in_overlap_func_;
};

} // namespace pcms

#endif // PCMS_COUPLER_OVERLAP_MASK_HPP
