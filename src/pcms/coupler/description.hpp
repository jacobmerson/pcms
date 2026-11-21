#ifndef PCMS_COUPLER_DESCRIPTION_HPP
#define PCMS_COUPLER_DESCRIPTION_HPP

#include "pcms/configuration.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/types.hpp"
#include <mpi.h>
#include <redev.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace pcms::detail
{

/// @brief The description of an application's functions, as key=value pairs.
using Description = std::map<std::string, std::string>;

/// @brief The description format version, stored under "pcms_schema".
inline constexpr std::string_view DescriptionSchema = "1";

/// @brief Writes `description` as "key=value" lines.
/// @throws pcms_error if a key contains '=' or an entry a line break
[[nodiscard]] std::string SerializeDescription(const Description& description);

/// @brief Reads the "key=value" lines of SerializeDescription.
/// @throws pcms_error if the schema is not DescriptionSchema
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

/// @brief Describes `chart` under `prefix`.
void DescribeChart(const CoordinateSystem& chart, const std::string& prefix,
                   Description& description);
/// @throws pcms_error if the chart kind is unknown
[[nodiscard]] std::shared_ptr<const CoordinateSystem> ChartFromDescription(
  const Description& description, const std::string& prefix);

/// @brief Describes `basis` under `prefix`.
void DescribeValueBasis(const ValueBasis& basis, const std::string& prefix,
                        Description& description);
[[nodiscard]] ValueBasis ValueBasisFromDescription(
  const Description& description, const std::string& prefix);

/// @brief Describes a uniform-grid geometry under `prefix`.
/// @return no data for the server ranks, or nullopt if `geometry` is not a
///         uniform grid
[[nodiscard]] std::optional<std::vector<std::vector<GO>>> DescribeUniformGrid(
  const Geometry& geometry, const std::string& prefix,
  Description& description);
[[nodiscard]] std::shared_ptr<const Geometry> RebuildUniformGrid(
  const Description& description, const std::string& prefix);

#ifdef PCMS_ENABLE_OMEGA_H
/// @brief Describes an Omega_h simplex mesh under `prefix`; collective over
///        `comm`.
/// @param partition the server partition that routes the owned cells
/// @return each server rank's owned cells with their vertices, or nullopt if
///         `geometry` is not an Omega_h mesh
/// @throws pcms_error if the mesh is not simplicial or a vertex has no owner
[[nodiscard]] std::optional<std::vector<std::vector<GO>>> DescribeOmegaH(
  const Geometry& geometry, const std::string& prefix, Description& description,
  MPI_Comm comm, redev::Partition partition, size_t num_servers);
/// @brief Rebuilds a mesh distributed over `comm` from the data of each
///        application rank.
[[nodiscard]] std::shared_ptr<const Geometry> RebuildOmegaH(
  const Description& description, const std::string& prefix,
  const std::vector<std::vector<GO>>& blocks, MPI_Comm comm);
#endif

} // namespace pcms::detail

#endif // PCMS_COUPLER_DESCRIPTION_HPP
