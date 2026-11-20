#ifndef PCMS_TRANSFER_OMEGA_H_INTERSECTION_QUADRATURE_HPP
#define PCMS_TRANSFER_OMEGA_H_INTERSECTION_QUADRATURE_HPP

#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/geometry/quadrature.hpp"
#include <memory>

namespace pcms::omega_h
{

// Quadrature of the pair of simplex meshes (triangles in 2D, tetrahedra in 3D):
// the reference rule of options.order on the sub-simplices that partition
// each target element's intersection with the source mesh (r3d clipping and
// star decomposition). Integrating a function over the target mesh as the
// sum of its integrals over the pieces that lie in single source elements
// is exact when the integrand is a polynomial of at most that order on each
// piece, e.g. the product of a source field and a target basis function of
// total degree `order`.
//
// The points are generated in target elements, so they are bound on the
// target geometry without a search (`points`); `other` names the source
// element each lies in, to bind the source geometry the same way.
//
// This lives next to the transfer operators rather than in the Omega_h
// backend because r3d, which does the clipping, is a dependency of
// pcms::transfer only.
//
// @throws pcms_error if a geometry is not on a Cartesian simplex mesh, the
//         meshes' dimensions differ, or MeshFields is not enabled
Quadrature CreateIntersectionQuadrature(
  std::shared_ptr<const OmegaHGeometry> source,
  std::shared_ptr<const OmegaHGeometry> target,
  const QuadratureOptions& options);

} // namespace pcms::omega_h

#endif // PCMS_TRANSFER_OMEGA_H_INTERSECTION_QUADRATURE_HPP
