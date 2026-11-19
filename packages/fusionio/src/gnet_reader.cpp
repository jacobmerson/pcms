#include "gnet_reader.hpp"
#include <hdf5.h>
#include <stdexcept>
#include <utility>

namespace fusionio::detail
{

namespace
{

// An open HDF5 object, closed by `Close`.
template <herr_t (*Close)(hid_t)>
class Handle
{
public:
  Handle(hid_t id, const std::string& what) : id_(id)
  {
    if (id_ < 0) {
      throw std::runtime_error("Error opening " + what);
    }
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() { Close(id_); }
  [[nodiscard]] hid_t Id() const noexcept { return id_; }

private:
  hid_t id_;
};

using File = Handle<H5Fclose>;
using Dataset = Handle<H5Dclose>;
using Dataspace = Handle<H5Sclose>;

std::vector<hsize_t> DatasetDims(hid_t file, const std::string& path)
{
  const Dataset dataset(H5Dopen(file, path.c_str(), H5P_DEFAULT),
                        "dataset: " + path);
  const Dataspace space(H5Dget_space(dataset.Id()), "dataspace of " + path);
  std::vector<hsize_t> dims(H5Sget_simple_extent_ndims(space.Id()));
  H5Sget_simple_extent_dims(space.Id(), dims.data(), nullptr);
  return dims;
}

void ReadDataset(hid_t file, const std::string& path, hid_t type, void* buffer)
{
  const Dataset dataset(H5Dopen(file, path.c_str(), H5P_DEFAULT),
                        "dataset: " + path);
  if (H5Dread(dataset.Id(), type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) < 0) {
    throw std::runtime_error("Error reading dataset: " + path);
  }
}

} // namespace

int GnetData::PoloidalOffset(int rho)
{
  // 4 (1 + ... + rho)
  return 2 * rho * (rho + 1);
}

std::size_t GnetData::NumCells() const
{
  return static_cast<std::size_t>(PoloidalOffset(num_rho)) * num_zeta;
}

std::size_t GnetData::NumMarkers() const
{
  return NumCells() * num_energy * num_pitch;
}

std::vector<double> GnetData::NumberDensity() const
{
  std::vector<double> density(f5d);
  const std::size_t per_cell = static_cast<std::size_t>(num_energy) * num_pitch;
  for (std::size_t cell = 0; cell < volume3d.size(); ++cell) {
    for (std::size_t k = 0; k < per_cell; ++k) {
      density[cell * per_cell + k] /= volume3d[cell];
    }
  }
  return density;
}

GnetData ReadGnet(const std::string& path)
{
  const File file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                  "HDF5 file: " + path);
  GnetData data;
  const std::pair<const char*, int*> dimensions[] = {
    {"mxenergy", &data.num_energy},
    {"mxrho", &data.num_rho},
    {"mxzeta", &data.num_zeta},
    {"mxpitch", &data.num_pitch}};
  for (const auto& [name, value] : dimensions) {
    ReadDataset(file.Id(), std::string("/distribution5D/dimensions/") + name,
                H5T_NATIVE_INT, value);
  }
  ReadDataset(file.Id(), "/distribution5D/range/engmax_eV", H5T_NATIVE_DOUBLE,
              &data.max_energy_ev);
  ReadDataset(file.Id(), "/distribution5D/range/zetamax_rad", H5T_NATIVE_DOUBLE,
              &data.max_zeta_rad);

  const std::string f5d = "/distribution5D/data/f5d";
  const auto f5d_dims = DatasetDims(file.Id(), f5d);
  if (f5d_dims.size() != 3) {
    throw std::runtime_error("Expected 3D dataset " + f5d + " in " + path);
  }
  if (f5d_dims[0] != data.NumCells() ||
      f5d_dims[1] != static_cast<hsize_t>(data.num_energy) ||
      f5d_dims[2] != static_cast<hsize_t>(data.num_pitch)) {
    throw std::runtime_error("f5d dimensions do not match the GNET grid in " +
                             path);
  }
  data.f5d.resize(f5d_dims[0] * f5d_dims[1] * f5d_dims[2]);
  ReadDataset(file.Id(), f5d, H5T_NATIVE_DOUBLE, data.f5d.data());

  const std::string volume = "/distribution5D/data/volume3d";
  const auto volume_dims = DatasetDims(file.Id(), volume);
  if (volume_dims.size() != 1 || volume_dims[0] != data.NumCells()) {
    throw std::runtime_error("Expected 1D dataset " + volume +
                             " with a value per cell in " + path);
  }
  data.volume3d.resize(volume_dims[0]);
  ReadDataset(file.Id(), volume, H5T_NATIVE_DOUBLE, data.volume3d.data());
  return data;
}

} // namespace fusionio::detail
