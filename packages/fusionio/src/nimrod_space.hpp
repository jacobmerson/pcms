#ifndef FUSIONIO_SRC_NIMROD_SPACE_HPP
#define FUSIONIO_SRC_NIMROD_SPACE_HPP

#include "fusionio/fusionio.hpp"
#include <memory>
#include <string>
#include <vector>

namespace fusionio::detail
{

/// A NIMROD dump opened for evaluation. Opaque: NIMROD is a private
/// dependency.
class NimrodSource;

/// Opens a NIMROD dump for evaluation of the fields named by the
/// single-character ids in `fields`, e.g. "bvn". NIMROD's evaluator holds one
/// dump per process, so this throws while another source is open; the
/// geometries, spaces and fields created from a source share and keep it
/// open. Only toroidal cases (geom='tor') are supported. Not thread safe: it
/// changes the current directory while NIMROD opens the dump.
///
/// @param case_dir directory holding the case's nimrod.in (default: the
/// current directory). It must be writable: NIMROD writes scratch files there
/// while opening the dump. `dump` is relative to the current directory.
std::shared_ptr<NimrodSource> OpenNimrodSource(
  const std::string& dump, const std::string& fields,
  const std::string& case_dir = "");

/// NIMROD's names for the quantities `source` can evaluate, e.g. "b" or "ti".
std::vector<std::string> AvailableQuantities(const NimrodSource& source);

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_NIMROD_SPACE_HPP
