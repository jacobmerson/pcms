#include "fusionio/gnet.hpp"
#include <hdf5.h>
#include <pcms/utility/assert.hpp>
#include <numbers>
#include <string>
#include <type_traits>

static_assert(std::is_same_v<hid_t, std::int64_t>,
              "GTCMarkerWriter stores HDF5 ids as std::int64_t");

namespace fusionio
{

namespace
{
constexpr hsize_t Columns = 6;
} // namespace

Domain GTCDomain(pcms::Real max_zeta)
{
  constexpr pcms::Real TwoPi = 2 * std::numbers::pi;
  return {{0.0, 1.0, Draw::Sqrt},
          {0.0, TwoPi},
          {0.0, max_zeta},
          {2000.0, 200000.0},
          {-1.0, 1.0}};
}

GTCMarkerWriter::GTCMarkerWriter(const std::string& path,
                                 std::size_t num_markers)
  : num_markers_(num_markers)
{
  file_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_ < 0) {
    throw pcms::pcms_error("Failed to create HDF5 file: " + path);
  }
  const hsize_t dims[2] = {num_markers, Columns};
  const hid_t space = H5Screate_simple(2, dims, nullptr);
  dataset_ = H5Dcreate(file_, "/gtc_n6d", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                       H5P_DEFAULT, H5P_DEFAULT);
  H5Sclose(space);
  if (dataset_ < 0) {
    H5Fclose(file_);
    throw pcms::pcms_error("Failed to create /gtc_n6d in " + path);
  }
}

GTCMarkerWriter::~GTCMarkerWriter()
{
  H5Dclose(dataset_);
  H5Fclose(file_);
}

void GTCMarkerWriter::Append(const MarkerRows& rows)
{
  if (rows.extent(1) != Columns) {
    throw pcms::pcms_error("GTCMarkerWriter: markers need 6 columns");
  }
  const std::size_t n = rows.extent(0);
  if (written_ + n > num_markers_) {
    throw pcms::pcms_error("GTCMarkerWriter: more markers than the file's " +
                           std::to_string(num_markers_));
  }
  if (n == 0) {
    return;
  }
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), rows);
  const hsize_t start[2] = {written_, 0};
  const hsize_t count[2] = {n, Columns};
  const hid_t file_space = H5Dget_space(dataset_);
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count,
                      nullptr);
  const hid_t memory_space = H5Screate_simple(2, count, nullptr);
  const herr_t status = H5Dwrite(dataset_, H5T_NATIVE_DOUBLE, memory_space,
                                 file_space, H5P_DEFAULT, host.data());
  H5Sclose(memory_space);
  H5Sclose(file_space);
  if (status < 0) {
    throw pcms::pcms_error("GTCMarkerWriter: could not write the markers");
  }
  written_ += n;
}

void WriteGTCMarkers(const std::string& path, const MarkerRows& markers)
{
  GTCMarkerWriter writer(path, markers.extent(0));
  writer.Append(markers);
}

} // namespace fusionio
