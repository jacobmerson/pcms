#ifndef FUSIONIO_SRC_M3DC1_NATIVE_HPP
#define FUSIONIO_SRC_M3DC1_NATIVE_HPP

// Private (not installed): M3D-C1 fields evaluated natively, by Kokkos
// kernels on PCMS's Omega_h geometry, instead of by Fusion-IO 1.x. Fusion-IO
// 1.x still reads the file (mesh and coefficients); the evaluation reproduces
// m3dc1_field::eval.

#include "fusionio/fusionio.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/memory_spaces.h"
#include <Kokkos_Core.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <memory>

class m3dc1_mesh;

namespace fusionio::detail
{

class FusionIOSource;

/// The (R, Z) mesh of an M3D-C1 file as an Omega_h triangle mesh, its PCMS
/// geometry (CylindricalRZ) and the element frames.
///
/// M3D-C1 stores each triangle as a frame: the origin (x, z), the rotation
/// (co, sn) of its local (xi, eta) axes and the extents a, b, c of its
/// vertices (-b, 0), (a, 0), (0, c) in them (m3dc1_mesh::local_to_global).
/// The mesh's vertices are those corners, merged within 1e-4 of the smallest
/// element's a + b + c (the frames are stored in single precision; 1.x's
/// m3dc1_mesh::shared_nodes uses the element's own bound); its elements are
/// the file's, in the file's order, so an element id of the geometry's search
/// is the file's element. Evaluation uses the frames, as 1.x does.
class M3DC1NativeMesh final
  : public std::enable_shared_from_this<M3DC1NativeMesh>
{
public:
  /// [element][x, z, co, sn, b]: what global_to_local needs
  using Frames = Kokkos::View<const pcms::Real**, Kokkos::LayoutRight,
                              pcms::DeviceMemorySpace>;

  /// Own it by std::shared_ptr (GetGeometry uses shared_from_this()).
  /// @param mesh a 2D (not stellarator) M3D-C1 mesh
  explicit M3DC1NativeMesh(const m3dc1_mesh& mesh);
  M3DC1NativeMesh(const M3DC1NativeMesh&) = delete;
  M3DC1NativeMesh& operator=(const M3DC1NativeMesh&) = delete;

  /// The geometry of the mesh, which keeps this object (and so the mesh it
  /// refers to) alive.
  [[nodiscard]] std::shared_ptr<const pcms::Geometry> GetGeometry() const;

  [[nodiscard]] const Frames& GetFrames() const noexcept { return frames_; }

  [[nodiscard]] pcms::LO NumElements() const noexcept
  {
    return static_cast<pcms::LO>(frames_.extent(0));
  }

  /// The toroidal period of the file's mesh (m3dc1_mesh::period).
  [[nodiscard]] pcms::Real Period() const noexcept { return period_; }

private:
  // declared in destruction order: the geometry refers to the mesh, and the
  // mesh to the library
  std::unique_ptr<Omega_h::Library> library_;
  std::unique_ptr<Omega_h::Mesh> mesh_;
  std::shared_ptr<const pcms::Geometry> geometry_;
  Frames frames_;
  pcms::Real period_;
};

/// An M3D-C1 source whose functions are evaluated natively: GetGeometry() is
/// the M3DC1NativeMesh's (R, Z) geometry, and CreateFunction creates the
/// axisymmetric (2D) fields of `source`: the scalar quantities Fusion-IO 1.x
/// reads as one stored field (poloidal flux and its normalized form,
/// pressures but the main ion's, densities, temperatures, resistivity,
/// radiation, bootstrap-current terms), with Gradient and Hessian.
/// Quantities, times, period and series are `wrapped`'s.
/// @param source an M3D-C1 source
/// @param wrapped the Fusion-IO 1.x source of `source`
/// @throws pcms_error if the file's mesh is not a 2D M3D-C1 mesh
std::shared_ptr<Source> CreateM3DC1NativeSource(
  std::shared_ptr<FusionIOSource> source, std::shared_ptr<Source> wrapped);

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_M3DC1_NATIVE_HPP
