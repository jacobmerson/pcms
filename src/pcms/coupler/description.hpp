#ifndef PCMS_COUPLER_DESCRIPTION_HPP
#define PCMS_COUPLER_DESCRIPTION_HPP

#include "pcms/configuration.hpp"
#include "pcms/field/field.hpp"
#include "pcms/field/field_metadata.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/types.hpp"
#include <mpi.h>
#include <redev.h>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pcms::detail
{

/// @brief The description of an application's functions, as key=value pairs.
using Description = std::map<std::string, std::string>;

/// @brief The coordinate systems of a description, numbered by identity.
using DescribedSystems = std::vector<std::shared_ptr<const CoordinateSystem>>;

/// @brief The description format version, stored under "pcms_schema".
inline constexpr std::string_view DescriptionSchema = "1";

/// @return an empty description with the schema and the PCMS version
[[nodiscard]] Description CreateDescription();

/// @brief Writes `description` as "key=value" lines.
/// @throws pcms_error if a key contains '=' or an entry a line break
[[nodiscard]] std::string SerializeDescription(const Description& description);

/// @brief Reads the "key=value" lines of SerializeDescription.
/// @throws pcms_error if the schema is not DescriptionSchema or the PCMS
///         version is not this one
[[nodiscard]] Description ParseDescription(std::string_view text);

/// @return the value of `key`
/// @throws pcms_error if `description` has no `key`
[[nodiscard]] const std::string& Require(const Description& description,
                                         const std::string& key);

/// @return `value` as the shortest text that parses back to it
[[nodiscard]] std::string FormatReal(Real value);
/// @throws pcms_error if `text` is not a real number
[[nodiscard]] Real ParseReal(std::string_view text);
/// @throws pcms_error if `text` is not an integer
[[nodiscard]] GO ParseInteger(std::string_view text);

/// @brief Describes `system` as "system.N" unless it is in `systems`.
/// @return N
std::size_t DescribeSystem(
  const std::shared_ptr<const CoordinateSystem>& system,
  DescribedSystems& systems, Description& description);
/// @return the systems "system.N" of `description`, by N
/// @throws pcms_error if a system kind is unknown
[[nodiscard]] DescribedSystems SystemsFromDescription(
  const Description& description);

/// @brief Describes `basis` under `prefix`.
void DescribeValueBasis(const ValueBasis& basis, const std::string& prefix,
                        DescribedSystems& systems, Description& description);
[[nodiscard]] ValueBasis ValueBasisFromDescription(
  const Description& description, const std::string& prefix,
  const DescribedSystems& systems);

/// @brief Describes `metadata` under `prefix`; units by their text.
void DescribeMetadata(const FieldMetadata& metadata, const std::string& prefix,
                      Description& description);
/// @throws pcms_error if the units do not parse
[[nodiscard]] FieldMetadata MetadataFromDescription(
  const Description& description, const std::string& prefix);

/// @brief Describes a uniform-grid geometry under `prefix`.
/// @return no data for the server ranks, or nullopt if `geometry` is not a
///         uniform grid
[[nodiscard]] std::optional<std::vector<std::vector<GO>>> DescribeUniformGrid(
  const Geometry& geometry, const std::string& prefix,
  DescribedSystems& systems, Description& description);
[[nodiscard]] std::shared_ptr<const Geometry> RebuildUniformGrid(
  const Description& description, const std::string& prefix,
  const DescribedSystems& systems);

/// @brief Describes an embedded geometry over a uniform grid under `prefix`:
///        the grid, the options, and the position and given metric, whose
///        values the first rank of `comm` sends to every server rank.
/// @return the values for each server rank, or nullopt if `geometry` is not
///         an embedded geometry
/// @throws pcms_error if the parameter geometry is not a uniform grid or the
///         position or metric is not a Lagrange Function of it
[[nodiscard]] std::optional<std::vector<std::vector<GO>>> DescribeEmbedded(
  const Geometry& geometry, const std::string& prefix,
  DescribedSystems& systems, Description& description, MPI_Comm comm,
  std::size_t num_servers);
/// @param blocks the data of each application rank
[[nodiscard]] std::shared_ptr<const Geometry> RebuildEmbedded(
  const Description& description, const std::string& prefix,
  const DescribedSystems& systems, const std::vector<std::vector<GO>>& blocks);

/// @return the Lagrange order of the layout of a described space
/// @throws pcms_error if no kind describes the space
[[nodiscard]] int LagrangeOrder(const FieldLayout& layout);

/// @brief Describes a Lagrange Function on a uniform grid, but not its values
void DescribeLagrangeFunction(const Function<Real>& function,
                              const std::string& key, DescribedSystems& systems,
                              Description& description);

/// @brief Rebuilds a Function described by DescribeLagrangeFunction.
/// @param values the DOF holders' values first; advanced past them
[[nodiscard]] std::shared_ptr<const Function<Real>> RebuildLagrangeFunction(
  const Description& description, const std::string& key,
  std::shared_ptr<const Geometry> geometry, const DescribedSystems& systems,
  std::span<const Real>& values);

#ifdef PCMS_ENABLE_OMEGA_H
/// @brief Describes an Omega_h simplex mesh under `prefix`; collective over
///        `comm`.
/// @param partition the server partition that routes the owned cells
/// @return each server rank's owned cells with their vertices, or nullopt if
///         `geometry` is not an Omega_h mesh
/// @throws pcms_error if the mesh is not simplicial or a vertex has no owner
[[nodiscard]] std::optional<std::vector<std::vector<GO>>> DescribeOmegaH(
  const Geometry& geometry, const std::string& prefix,
  DescribedSystems& systems, Description& description, MPI_Comm comm,
  redev::Partition partition, size_t num_servers);
/// @brief Rebuilds a mesh distributed over `comm` from the data of each
///        application rank.
[[nodiscard]] std::shared_ptr<const Geometry> RebuildOmegaH(
  const Description& description, const std::string& prefix,
  const DescribedSystems& systems, const std::vector<std::vector<GO>>& blocks,
  MPI_Comm comm);
#endif

} // namespace pcms::detail

#endif // PCMS_COUPLER_DESCRIPTION_HPP
