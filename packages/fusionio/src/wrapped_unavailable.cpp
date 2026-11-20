#include "fusionio/fusionio.hpp"
#include "fusionio/geqdsk.hpp"
#include <pcms/utility/assert.hpp>

namespace fusionio
{

std::shared_ptr<Source> Open(Format format, const std::string& path,
                             Evaluation evaluation)
{
  if (format == Format::GEQDSK && evaluation == Evaluation::Native) {
    return OpenGeqdsk(path);
  }
  throw pcms::pcms_error(
    "fusionio::Open: Fusion-IO was built without Fusion-IO 1.x "
    "(FUSIONIO_ENABLE_WRAPPED)");
}

} // namespace fusionio
