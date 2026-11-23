#include "fusionio/bout.hpp"
#include <pcms/utility/assert.hpp>

namespace fusionio
{

namespace
{

[[noreturn]] void Unavailable()
{
  throw pcms::pcms_error("fusionio: Fusion-IO was built without BOUT++ grid "
                         "files (FUSIONIO_ENABLE_BOUT)");
}

} // namespace

BoutGridFile ReadBoutGrid(const std::string& /*path*/, int /*nz*/,
                          int /*zperiod*/)
{
  Unavailable();
}

std::shared_ptr<Source> OpenBout(const std::string& /*path*/, int /*nz*/,
                                 int /*zperiod*/)
{
  Unavailable();
}

} // namespace fusionio
