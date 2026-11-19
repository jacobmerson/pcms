#ifndef PCMS_BACKENDS_OMEGA_H_GEOMETRY_HPP
#define PCMS_BACKENDS_OMEGA_H_GEOMETRY_HPP

#include "pcms/geometry/geometry.hpp"
#include <Omega_h_array.hpp>
#include <Omega_h_defines.hpp>
#include <string>

namespace pcms::omega_h
{

/// How an Omega_h geometry finds the elements containing points.
enum class OmegaHSearchMethod
{
  /// a uniform grid of candidate elements
  Grid,
  /// an ArborX bounding volume hierarchy over the elements; needs a PCMS built
  /// with PCMS_ENABLE_ARBORX (creating the geometry throws otherwise). Same
  /// records as Grid for every point in the mesh; SearchOptions::divisions is
  /// not used.
  BoundingVolumeHierarchy
};

/// Options of an Omega_h geometry.
struct OmegaHGeometryOptions
{
  /// divisions and tolerance of the containing-element search
  SearchOptions search;
  /// the data structure of the containing-element search
  OmegaHSearchMethod search_method = OmegaHSearchMethod::Grid;
  /// the tag holding global entity ids
  std::string global_id_name = "global";
  /// per-vertex ownership override: a vertex the mask is 0 for is not owned,
  /// whatever the mesh says; empty means the mesh's own ownership. Layouts
  /// whose DOF holders are the mesh vertices apply it; layouts of other
  /// entities do not.
  Omega_h::Read<Omega_h::I8> owned_mask;
};

} // namespace pcms::omega_h

#endif // PCMS_BACKENDS_OMEGA_H_GEOMETRY_HPP
