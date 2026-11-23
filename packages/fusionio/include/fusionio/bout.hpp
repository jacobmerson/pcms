#ifndef FUSIONIO_BOUT_HPP
#define FUSIONIO_BOUT_HPP

/// BOUT++ grids: the geometry and fields of one block of a BOUT++ grid as
/// PCMS objects, built from the arrays BOUT++ holds.

#include "fusionio/fusionio.hpp"
#include <pcms/field/embedding.hpp>
#include <pcms/field/field.hpp>
#include <pcms/field/function_space.hpp>
#include <pcms/utility/arrays.hpp>
#include <pcms/utility/memory_spaces.hpp>
#include <array>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace fusionio
{

using HostArray2D = pcms::Rank2View<const Real, pcms::HostMemorySpace>;
using HostArray1D = pcms::Rank1View<const Real, pcms::HostMemorySpace>;

/// @brief One block of a BOUT++ grid in field-aligned coordinates: the points
///        (x, y, z) at phi = z + zShift(x, y), interior and guard cells.
struct BoutGrid
{
  /// points of the block, guard cells included in x and y (BOUT++'s
  /// LocalNx, LocalNy, LocalNz)
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int mxg = 0;
  int myg = 0;
  /// the global x index of the block's first point and the global y index
  /// of its first interior point
  int x_offset = 0;
  int y_offset = 0;
  /// z covers 2 pi / zperiod
  int zperiod = 1;
  /// the separatrix indices; the defaults are a grid of closed field lines
  int ixseps1 = std::numeric_limits<int>::max();
  int ixseps2 = std::numeric_limits<int>::max();
  int jyseps1_1 = -1;
  int jyseps1_2 = 0;
  int jyseps2_1 = 0;
  int jyseps2_2 = std::numeric_limits<int>::max();
  /// [nx][ny]
  HostArray2D rxy;
  HostArray2D zxy;
  HostArray2D zshift;
  HostArray2D dx;
  HostArray2D dy;
  /// [nx] the change of zShift once around a closed field line
  HostArray1D shift_angle;
  Real dz = 0;
  /// g^xx, g^yy, g^zz, g^xy, g^xz, g^yz [nx][ny]
  std::array<HostArray2D, 6> g_contra;
};

/// @brief The geometry and the space of a BOUT++ grid's fields.
struct BoutSpaces
{
  /// the grid in (R, phi, Z) (pcms CylindricalRThetaZ(2 pi / zperiod)) over
  /// global index coordinates (x, y, z) in a parameter system: y is
  /// identified by the twist-shift of z by ShiftAngle / dz and z is periodic
  std::shared_ptr<const pcms::EmbeddedGeometry> geometry;
  /// multilinear in the index coordinates, a DOF holder per interior point,
  /// per point of the y = ny plane (the twist-shifted y = 0 plane) and per
  /// point of the z = nz plane (the z = 0 plane)
  std::shared_ptr<const pcms::FunctionSpace> space;
};

/// @brief Builds the geometry and space of a block.
/// @throws pcms_error if the grid has open field lines, is not one serial
///         block (offsets 0), has fewer than 2 interior points in x or y,
///         arrays of the wrong extents, or nz dz is not 2 pi / zperiod
[[nodiscard]] BoutSpaces Build(const BoutGrid& grid);

/// @brief The offset in BOUT++'s field array of the block ([nx][ny][nz])
///        of each DOF holder of `space`: the y = ny plane reads the first
///        upper y guard plane, the z = nz plane the z = 0 plane.
/// @param space Build(grid).space or a space on the same geometry
/// @throws pcms_error if the grid has no y guard cells or a DOF holder is
///         not a point of the block
[[nodiscard]] std::vector<pcms::LO> InteriorOffsets(
  const pcms::FunctionSpace& space, const BoutGrid& grid);

/// @brief Sets a scalar function's values from a BOUT++ field array.
/// @param offsets InteriorOffsets of the function's space
void CopyIn(std::span<const Real> values, std::span<const pcms::LO> offsets,
            pcms::Function<Real>& function);

/// @brief Writes a scalar function's values into a BOUT++ field array.
/// @param offsets InteriorOffsets of the function's space
void CopyOut(const pcms::Function<Real>& function,
             std::span<const pcms::LO> offsets, std::span<Real> values);

} // namespace fusionio

#endif // FUSIONIO_BOUT_HPP
