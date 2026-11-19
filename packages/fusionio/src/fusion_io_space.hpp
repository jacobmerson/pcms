#ifndef FUSIONIO_SRC_FUSION_IO_SPACE_HPP
#define FUSIONIO_SRC_FUSION_IO_SPACE_HPP

#include "fusionio/fusionio.hpp"
#include "pcms/field/function_space.hpp"
#include "pcms/geometry/geometry.hpp"
#include <memory>
#include <string>

namespace fusionio::detail
{

/// An open Fusion-IO file. Opaque: Fusion-IO is a private dependency.
class FusionIOSource;

/// Opens a Fusion-IO file, e.g. an M3D-C1 C1.h5 or a geqdsk file. Only
/// sources with cylindrical geometry are supported. The geometries, spaces
/// and fields created from it share and keep it open.
std::shared_ptr<FusionIOSource> OpenFusionIOSource(Format format,
                                                   const std::string& filename);

/// The geometry of an open Fusion-IO 1.x file: (R, phi, Z) points, which is
/// CylindricalRThetaZ (Fusion-IO's (R, phi, Z) is right-handed, so phi is
/// theta), no entities PCMS can see, and the file's own point location as the
/// search. Every quantity of the file shares its domain, so one
/// binding serves them all.
///
/// Location at bind (and again on Geometry::Rebind): M3D-C1 sources search
/// the domain probe's mesh (m3dc1_mesh::in_element, see
/// FusionIOSource::DomainMesh) under every policy, so OutOfBoundsMode::ERROR
/// throws at bind for points outside; other formats evaluate the domain
/// probe under OutOfBoundsMode::FILL only, and under ERROR evaluation reports
/// the points outside. OutOfBoundsMode::NEAREST_BOUNDARY is not supported.
/// OwnerHints are ignored: the file's mesh is not a PCMS geometry's.
class FusionIOGeometry final : public pcms::Geometry
{
public:
  explicit FusionIOGeometry(std::shared_ptr<FusionIOSource> source);

  [[nodiscard]] const std::shared_ptr<const pcms::CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return system_;
  }

  [[nodiscard]] const std::shared_ptr<FusionIOSource>& GetSource()
    const noexcept
  {
    return source_;
  }

protected:
  [[nodiscard]] std::shared_ptr<pcms::BoundPoints> BindImpl(
    const pcms::BindRequest& request) const override;

private:
  std::shared_ptr<FusionIOSource> source_;
  std::shared_ptr<const pcms::CoordinateSystem> system_;
};

/// The space of the quantities of one representation of a Fusion-IO source:
/// `num_components` components (1 for scalars, 3 for vectors) and, if
/// `derivatives`, Fusion-IO's derivatives (see ProvidesDerivatives), whose
/// evaluators then support DifferentialOperator::Gradient: the gradient of
/// scalar quantities as physical cylindrical components (d/dR, (1/R) d/dphi,
/// d/dZ). PCMS holds none of the values: its functions are created by
/// Source::CreateFunction and can be evaluated and used as transfer sources,
/// but not read or written.
std::shared_ptr<const pcms::FunctionSpace> CreateFusionIOSpace(
  std::shared_ptr<const FusionIOGeometry> geometry, int num_components,
  bool derivatives);

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_FUSION_IO_SPACE_HPP
