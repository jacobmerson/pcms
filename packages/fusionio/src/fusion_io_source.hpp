#ifndef FUSIONIO_SRC_FUSION_IO_SOURCE_HPP
#define FUSIONIO_SRC_FUSION_IO_SOURCE_HPP

// Private (not installed): the definition of the opaque FusionIOSource, for
// the Fusion-IO function space and its tests.

#include "fusionio/fusionio.hpp"
#include <fusion_io.h>
#include <memory>
#include <optional>
#include <vector>

namespace fusionio::detail
{

/// Closes a Fusion-IO source. Unlike deleting it, fio_close_source also closes
/// the file, which some sources (M3D-C1) do not do in their destructor.
struct FusionIOClose
{
  void operator()(fio_source* source) const { fio_close_source(&source); }
};

using FusionIOSourcePtr = std::unique_ptr<fio_source, FusionIOClose>;
using FusionIOFieldPtr = std::unique_ptr<fio_field>;

/// Fusion-IO's field type for a quantity.
[[nodiscard]] field_type FieldTypeOf(Quantity quantity);

/// 3 for vector quantities, 1 for scalars.
[[nodiscard]] int NumComponents(Quantity quantity);

/// An open Fusion-IO source and the quantities it provides.
class FusionIOSource
{
public:
  /// @param source must have cylindrical geometry
  explicit FusionIOSource(FusionIOSourcePtr source);

  /// Fusion-IO's interface is not const-correct, so this is shallow const.
  [[nodiscard]] fio_source& Get() const noexcept { return *source_; }

  /// Reads a quantity of the source; throws if Fusion-IO cannot.
  [[nodiscard]] FusionIOFieldPtr GetField(Quantity quantity,
                                          const fio_option_list& options) const;

  /// The quantities the source lists, in Quantity order.
  [[nodiscard]] const std::vector<Quantity>& Quantities() const noexcept
  {
    return quantities_;
  }

  [[nodiscard]] bool Provides(Quantity quantity) const;

  /// A field of the source (a scalar if it has one) whose evaluation tells
  /// whether a point is inside the source's domain. Read on first use.
  [[nodiscard]] fio_field& DomainProbe() const;

  /// For M3D-C1 sources, the mesh DomainProbe() is evaluated on: a point is
  /// inside the domain if m3dc1_mesh::in_element finds it, the search the
  /// probe's evaluation starts with. Null for other formats and for
  /// stellarator meshes (whose fields map the coordinates first). Read on
  /// first use; the source owns it.
  [[nodiscard]] m3dc1_mesh* DomainMesh() const;

  /// Search hints for `num_points` query points: sizeof_search_hint() bytes
  /// per point. Consecutive bindings to the same number of points share them,
  /// so point i starts where point i of the previous binding was found (the
  /// warm start lockstep tracers need). Hints are only first guesses, so
  /// unrelated bindings sharing them lose speed, never correctness.
  [[nodiscard]] std::shared_ptr<std::vector<char>> SearchHints(
    size_t num_points) const;

private:
  FusionIOSourcePtr source_;
  std::vector<Quantity> quantities_;
  mutable FusionIOFieldPtr probe_;
  mutable std::optional<m3dc1_mesh*> domain_mesh_;
  mutable std::shared_ptr<std::vector<char>> hints_;
};

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_FUSION_IO_SOURCE_HPP
