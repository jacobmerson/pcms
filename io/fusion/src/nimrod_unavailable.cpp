#include "fusion_io/fusion_io.hpp"
#include <pcms/utility/assert.h>

namespace fio
{

std::shared_ptr<Source> OpenNimrod(const std::string& /*dump*/,
                                   const std::string& /*fields*/,
                                   const std::string& /*case_dir*/)
{
  throw pcms::pcms_error("fio::OpenNimrod: Fusion-IO was built without NIMROD "
                         "(FUSION_IO_ENABLE_NIMROD)");
}

} // namespace fio
