#include "gnet_synthetic.hpp"
#include <hdf5.h>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace fusionio::test
{

namespace
{

template <herr_t (*Close)(hid_t)>
class Handle
{
public:
  explicit Handle(hid_t id) : id_(id)
  {
    if (id_ < 0) {
      throw std::runtime_error("synthetic GNET file: HDF5 call failed");
    }
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() { Close(id_); }
  [[nodiscard]] hid_t Id() const noexcept { return id_; }

private:
  hid_t id_;
};

// Writes a dataset of `dims` (and its groups) from `values`.
void Write(hid_t file, const char* path, hid_t type,
           const std::vector<hsize_t>& dims, const void* values)
{
  const Handle<H5Pclose> groups(H5Pcreate(H5P_LINK_CREATE));
  H5Pset_create_intermediate_group(groups.Id(), 1);
  const Handle<H5Sclose> space(
    H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr));
  const Handle<H5Dclose> dataset(H5Dcreate(
    file, path, type, space.Id(), groups.Id(), H5P_DEFAULT, H5P_DEFAULT));
  if (H5Dwrite(dataset.Id(), type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) < 0) {
    throw std::runtime_error("synthetic GNET file: cannot write dataset");
  }
}

void WriteScalar(hid_t file, const char* path, hid_t type, const void* value)
{
  Write(file, path, type, {1}, value);
}

} // namespace

double SyntheticNumberDensity(double rho, double theta, double zeta,
                              double energy, double pitch)
{
  return 40.0 * (1.0 - 0.6 * rho) * (1.0 + 0.3 * std::cos(theta)) *
         (1.0 + 0.2 * std::cos(5.0 * zeta)) *
         std::exp(-2.0 * energy / SyntheticMaxEnergyEv) *
         (1.0 + 0.5 * pitch * pitch);
}

double SyntheticVolume(double rho, double theta)
{
  return 0.1 * rho * (1.0 + 0.2 * std::sin(theta));
}

void WriteSyntheticGnet(const std::string& path, const GnetShape& shape)
{
  const Handle<H5Fclose> file(
    H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
  const std::pair<const char*, const int*> dimensions[] = {
    {"/distribution5D/dimensions/mxenergy", &shape.num_energy},
    {"/distribution5D/dimensions/mxrho", &shape.num_rho},
    {"/distribution5D/dimensions/mxzeta", &shape.num_zeta},
    {"/distribution5D/dimensions/mxpitch", &shape.num_pitch}};
  for (const auto& [name, value] : dimensions) {
    WriteScalar(file.Id(), name, H5T_NATIVE_INT, value);
  }
  WriteScalar(file.Id(), "/distribution5D/range/engmax_eV", H5T_NATIVE_DOUBLE,
              &SyntheticMaxEnergyEv);
  WriteScalar(file.Id(), "/distribution5D/range/zetamax_rad", H5T_NATIVE_DOUBLE,
              &SyntheticMaxZetaRad);

  std::vector<double> f5d;
  std::vector<double> volume;
  for (int rho = 0; rho < shape.num_rho; ++rho) {
    const double rho_c = (0.5 + rho) / shape.num_rho;
    const int num_theta = 4 * (rho + 1);
    for (int theta = 0; theta < num_theta; ++theta) {
      const double theta_c = (theta + 0.5) * 2.0 * std::numbers::pi / num_theta;
      for (int zeta = 0; zeta < shape.num_zeta; ++zeta) {
        const double zeta_c =
          (zeta + 0.5) * SyntheticMaxZetaRad / shape.num_zeta;
        const double v = SyntheticVolume(rho_c, theta_c);
        volume.push_back(v);
        for (int energy = 0; energy < shape.num_energy; ++energy) {
          const double energy_c =
            SyntheticMaxEnergyEv * (energy + 0.5) / shape.num_energy;
          for (int pitch = 0; pitch < shape.num_pitch; ++pitch) {
            const double pitch_c = 2.0 * (pitch + 0.5) / shape.num_pitch - 1.0;
            f5d.push_back(v * SyntheticNumberDensity(rho_c, theta_c, zeta_c,
                                                     energy_c, pitch_c));
          }
        }
      }
    }
  }
  Write(file.Id(), "/distribution5D/data/f5d", H5T_NATIVE_DOUBLE,
        {volume.size(), static_cast<hsize_t>(shape.num_energy),
         static_cast<hsize_t>(shape.num_pitch)},
        f5d.data());
  Write(file.Id(), "/distribution5D/data/volume3d", H5T_NATIVE_DOUBLE,
        {volume.size()}, volume.data());
}

} // namespace fusionio::test
