#ifndef FUSIONIO_BOUT_POINTS_HPP
#define FUSIONIO_BOUT_POINTS_HPP

#include "fusionio/bout.hpp"
#include <vector>

namespace fusionio::detail
{

/// @brief A DOF holder as a point of a BOUT++ block: local x, local y of the
///        point or, on the y = ny plane, of the y = 0 plane, z (nz on the
///        z = nz plane), and whether it is on the y = ny plane.
struct BlockPoint
{
  int ix;
  int jy;
  int kz;
  bool seam;
};

/// @brief The DOF holders of `space` as points of the block.
/// @throws pcms_error if a DOF holder is not a point of the block
[[nodiscard]] std::vector<BlockPoint> BlockPoints(
  const pcms::FunctionSpace& space, const BoutGrid& grid);

} // namespace fusionio::detail

#endif // FUSIONIO_BOUT_POINTS_HPP
