#ifndef PCMS_COUPLER_DESCRIPTION_HPP
#define PCMS_COUPLER_DESCRIPTION_HPP

// Private: the descriptions an application sends the server in
// ApplicationComm::Setup, and the registry of the kinds of function spaces
// that can be described.

#include "pcms/configuration.hpp"
#include "pcms/field/function_space.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/types.hpp"
#include <mpi.h>
#include <redev.h>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pcms::detail
{

/// The description of the functions an application sends during setup: flat
/// key/value pairs, written as "key=value" lines so that it fits an ADIOS2
/// string attribute. Keys are dotted paths (geometry.0.kind, ...).
using Description = std::map<std::string, std::string>;

/// Version of the description format, stored under "pcms_schema". A server
/// rejects descriptions of another version.
inline constexpr std::string_view DescriptionSchema = "1";

[[nodiscard]] std::string SerializeDescription(const Description& description);

/// @throws pcms_error if a line has no '=' or the schema is not
///         DescriptionSchema
[[nodiscard]] Description ParseDescription(std::string_view text);

/// @throws pcms_error naming `key` if it is missing
[[nodiscard]] const std::string& Require(const Description& description,
                                         const std::string& key);

/// A real number written so that parsing it gives the same value.
[[nodiscard]] std::string FormatReal(Real value);
/// @throws pcms_error if `text` is not a real number
[[nodiscard]] Real ParseReal(std::string_view text);
/// @throws pcms_error if `text` is not an integer
[[nodiscard]] GO ParseInteger(std::string_view text);

/// Describes a chart (coordinate system) under `prefix`.
void DescribeChart(const CoordinateSystem& chart, const std::string& prefix,
                   Description& description);
/// @throws pcms_error for a chart kind the server does not know
[[nodiscard]] std::shared_ptr<const CoordinateSystem> ChartFromDescription(
  const Description& description, const std::string& prefix);

/// Describes a value basis under `prefix`.
void DescribeValueBasis(const ValueBasis& basis, const std::string& prefix,
                        Description& description);
[[nodiscard]] ValueBasis ValueBasisFromDescription(
  const Description& description, const std::string& prefix);

/// The coordinates of the DOF holders of `layout` (its owned ones), on the
/// host.
[[nodiscard]] Kokkos::View<Real**, Kokkos::LayoutRight, HostMemorySpace>
HostDOFHolderCoordinates(const FieldLayout& layout);

/// The server rank `partition` assigns an entity with classification
/// (class_dim, class_id) at `coord` (dim coordinates) to.
[[nodiscard]] int RankOf(const redev::Partition& partition, LO class_dim,
                         LO class_id, const Real* coord, int dim);

/// `partition` for routing points of dimension `dim`. A client receives an
/// RCB partition without its dimension, so it is rebuilt with `dim`.
[[nodiscard]] redev::Partition RoutingPartition(
  const redev::Partition& partition, int dim);

/// What both sides of a setup know about their own and the other side.
struct SetupContext
{
  MPI_Comm comm;
  int rank;
  int size;
  /// number of ranks of the other side (server ranks on a client, application
  /// ranks on the server)
  int peer_size;
  const redev::Partition& partition;
};

/// The setup data of one geometry one application rank sends one server
/// rank.
struct SetupBlock
{
  std::vector<GO> ints;
  std::vector<Real> reals;
};

/// The setup data of one geometry a server rank received from one
/// application rank.
struct ReceivedBlock
{
  std::span<const GO> ints;
  std::span<const Real> reals;
};

/// A DOF holder a server rank receives the values of: its owned index in the
/// server's layout, the application rank that sends them and the holder's
/// owned index in that rank's layout.
struct Claim
{
  LO holder;
  LO app_rank;
  LO app_index;
};

/// A geometry rebuilt on the server, and the DOF holders this server rank
/// receives for a layout on it.
struct RebuiltGeometry
{
  std::shared_ptr<const Geometry> geometry;
  std::function<std::vector<Claim>(const FieldLayout&)> claims;
};

/// A kind of function space an application can describe: how a client
/// describes its geometry and space, and how the server rebuilds them. Adding
/// a kind is adding an implementation to SpaceKinds().
class SpaceKind
{
public:
  virtual ~SpaceKind() noexcept = default;

  /// the kind's name, the value of geometry.<i>.kind
  [[nodiscard]] virtual std::string_view Name() const noexcept = 0;

  /// @return whether this kind describes `space`
  [[nodiscard]] virtual bool Describes(const FunctionSpace& space) const = 0;

  /// Describes the geometry of a space Describes() accepts under `prefix`.
  /// @return the geometry's block for each server rank
  [[nodiscard]] virtual std::vector<SetupBlock> DescribeGeometry(
    const Geometry& geometry, const std::string& prefix,
    Description& description, const SetupContext& context) const = 0;

  /// Describes a space Describes() accepts under `prefix`.
  virtual void DescribeSpace(const FunctionSpace& space,
                             const std::string& prefix,
                             Description& description) const = 0;

  /// Rebuilds a geometry DescribeGeometry described.
  /// @param blocks the geometry's block from each application rank
  [[nodiscard]] virtual RebuiltGeometry RebuildGeometry(
    const Description& description, const std::string& prefix,
    std::span<const ReceivedBlock> blocks,
    const SetupContext& context) const = 0;

  /// Rebuilds a space DescribeSpace described on `geometry`.
  [[nodiscard]] virtual std::shared_ptr<const FunctionSpace> RebuildSpace(
    const RebuiltGeometry& geometry, const Description& description,
    const std::string& prefix) const = 0;
};

/// The kinds of spaces an application can describe.
[[nodiscard]] const std::vector<const SpaceKind*>& SpaceKinds();

/// @throws pcms_error if no kind describes `space`
[[nodiscard]] const SpaceKind& FindSpaceKind(const FunctionSpace& space);

/// @throws pcms_error if there is no kind called `name`
[[nodiscard]] const SpaceKind& FindSpaceKind(std::string_view name);

/// The uniform-grid Lagrange spaces of order 0 and 1.
[[nodiscard]] const SpaceKind& UniformGridSpaceKind();
#ifdef PCMS_ENABLE_OMEGA_H
/// The Lagrange spaces of order 1 on Omega_h simplex meshes.
[[nodiscard]] const SpaceKind& OmegaHSpaceKind();
#endif

} // namespace pcms::detail

#endif // PCMS_COUPLER_DESCRIPTION_HPP
