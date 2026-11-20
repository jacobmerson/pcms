#include "geqdsk_reader.hpp"
#include <pcms/utility/assert.hpp>
#include <fstream>
#include <sstream>

namespace fusionio::detail
{

GeqdskData ReadGeqdsk(const std::string& path)
{
  std::ifstream file(path);
  if (!file) {
    throw pcms::pcms_error("ReadGeqdsk: cannot open '" + path + "'");
  }
  const auto fail = [&](const std::string& what) {
    return pcms::pcms_error("ReadGeqdsk: '" + path + "': " + what);
  };

  // The first line is a description followed by an id and the grid's sizes
  // as its last two numbers.
  GeqdskData data;
  std::string header;
  std::getline(file, header);
  std::istringstream words(header);
  std::vector<std::string> tokens;
  for (std::string word; words >> word;) {
    tokens.push_back(word);
  }
  try {
    if (tokens.size() < 3) {
      throw std::invalid_argument("header");
    }
    data.nw = std::stoi(tokens[tokens.size() - 2]);
    data.nh = std::stoi(tokens[tokens.size() - 1]);
  } catch (const std::logic_error&) {
    throw fail("the first line does not end in the grid's sizes");
  }
  // a cubic spline needs four points
  if (data.nw < 4 || data.nh < 4) {
    throw fail("the grid is smaller than 4 x 4");
  }

  // Fortran numbers may run together ("1.0E+00-2.0E+00"), which operator>>
  // reads as two.
  const auto number = [&]() {
    Real value = 0;
    if (!(file >> value)) {
      throw fail("the data ends early or holds a non-number");
    }
    return value;
  };
  const auto numbers = [&](size_t n) {
    std::vector<Real> values(n);
    for (Real& value : values) {
      value = number();
    }
    return values;
  };

  data.rdim = number();
  data.zdim = number();
  number(); // rcentr
  data.rleft = number();
  data.zmid = number();
  data.rmaxis = number();
  data.zmaxis = number();
  data.simag = number();
  data.sibry = number();
  numbers(11); // bcentr, then the two lines repeating current, simag, rmaxis,
               // zmaxis and sibry between unused values
  const auto size = static_cast<size_t>(data.nw);
  data.fpol = numbers(size);
  data.pres = numbers(size);
  numbers(2 * size); // ffprim, pprime
  data.psirz = numbers(size * static_cast<size_t>(data.nh));
  data.qpsi = numbers(size);
  if (data.rdim <= 0 || data.zdim <= 0 || data.simag == data.sibry) {
    throw fail("the grid or the flux range is empty");
  }

  // The curves are optional: files without them end after q.
  int nbdry = 0, nlim = 0;
  if (file >> nbdry >> nlim) {
    if (nbdry < 0 || nlim < 0) {
      throw fail("negative number of boundary or limiter points");
    }
    const auto curve = [&](int n) {
      Polyline points(static_cast<size_t>(n));
      for (auto& point : points) {
        point[0] = number();
        point[1] = number();
      }
      return points;
    };
    data.boundary = curve(nbdry);
    data.limiter = curve(nlim);
  }
  return data;
}

} // namespace fusionio::detail
