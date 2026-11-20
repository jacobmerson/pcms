#ifndef FUSIONIO_SRC_GEQDSK_READER_HPP
#define FUSIONIO_SRC_GEQDSK_READER_HPP

// Private (not installed): the contents of a geqdsk file.

#include "fusionio/geqdsk.hpp"
#include <string>
#include <vector>

namespace fusionio::detail
{

/// A geqdsk file (G-EQDSK, EFIT's format), in the file's units.
struct GeqdskData
{
  /// points of the (R, Z) grid
  int nw = 0, nh = 0;
  /// extent of the grid: R from rleft over rdim, Z from zmid - zdim / 2 over
  /// zdim
  Real rdim = 0, zdim = 0, rleft = 0, zmid = 0;
  /// the magnetic axis and the psi at it and at the boundary (Wb/rad)
  Real rmaxis = 0, zmaxis = 0, simag = 0, sibry = 0;
  /// F, p and q on the uniform grid of nw values from simag to sibry
  std::vector<Real> fpol, pres, qpsi;
  /// psi on the grid, R fastest
  std::vector<Real> psirz;
  Polyline boundary, limiter;
};

/// Reads a geqdsk file.
/// @throws pcms_error if it cannot be read or is malformed
GeqdskData ReadGeqdsk(const std::string& path);

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_GEQDSK_READER_HPP
