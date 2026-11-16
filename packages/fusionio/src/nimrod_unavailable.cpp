#include "fusionio/fusionio.hpp"
#include <pcms/utility/assert.h>

namespace fusionio
{

std::shared_ptr<Source> OpenNimrod(const std::string& /*dump*/,
                                   const std::string& /*fields*/,
                                   const std::string& /*case_dir*/)
{
  throw pcms::pcms_error(
    "fusionio::OpenNimrod: Fusion-IO was built without NIMROD "
    "(FUSIONIO_ENABLE_NIMROD)");
}

} // namespace fusionio
