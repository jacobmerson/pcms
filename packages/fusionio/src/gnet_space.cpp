#include "fusionio/gnet.hpp"
#include "gnet_reader.hpp"
#include "gnet_search.hpp"
#include <pcms/backends/point_cloud/geometry.hpp>
#include <pcms/field/field_layout.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/field/mls_options.hpp>
#include <pcms/geometry/geometry.hpp>
#include <pcms/utility/arrays.hpp>
#include <pcms/utility/assert.hpp>
#include <pcms/utility/coordinate_systems/parameter.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace fusionio
{

namespace
{

using pcms::DeviceMemorySpace;
using pcms::HostMemorySpace;
using pcms::LO;
using pcms::Real;

constexpr int Dim = 5;
constexpr Real TwoPi = 2 * std::numbers::pi;

// The supports (GNET's bin search) of every query point, count^5 each.
void SearchBins(pcms::Rank2View<const Real, DeviceMemorySpace> queries,
                detail::GnetGrid grid, int count,
                Kokkos::View<LO*, DeviceMemorySpace> ptr,
                Kokkos::View<LO*, DeviceMemorySpace> idx)
{
  const LO n = static_cast<LO>(queries.extent(0));
  const LO per_query = count * count * count * count * count;
  Kokkos::parallel_for(
    "fusionio search gnet bins",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const LO i) {
      double query[Dim];
      for (int d = 0; d < Dim; ++d) {
        query[d] = queries(i, d);
      }
      ptr(i + 1) = (i + 1) * per_query;
      detail::GnetSupports(grid, count, query, &idx(i * per_query));
    });
}

// The bins are the supports of the 5D query points: GNET's own search, as the
// Neighbourhood of a geometry kind that PCMS's point cloud delegates to (the
// supports are the sites' indices, the markers' in the file).
class GnetGeometry final : public pcms::Geometry
{
public:
  GnetGeometry(std::shared_ptr<const pcms::CoordinateSystem> system,
               detail::GnetGrid grid, int count)
    : system_(std::move(system)), grid_(grid), count_(count)
  {
  }

  const std::shared_ptr<const pcms::CoordinateSystem>& GetCoordinateSystem()
    const noexcept override
  {
    return system_;
  }

  pcms::SupportResults Neighbourhood(
    pcms::CoordinateView<DeviceMemorySpace> coords, int /*site_dim*/,
    const pcms::NeighbourhoodOptions& /*options*/,
    const pcms::OwnerHints& /*owners*/ = {}) const override
  {
    const auto queries = coords.GetValues();
    const std::size_t n = queries.extent(0);
    const std::size_t per_query =
      static_cast<std::size_t>(count_) * count_ * count_ * count_ * count_;
    Kokkos::View<LO*, DeviceMemorySpace> ptr("gnet supports ptr", n + 1);
    Kokkos::View<LO*, DeviceMemorySpace> idx(
      Kokkos::view_alloc("gnet supports", Kokkos::WithoutInitializing),
      n * per_query);
    // the radius is not used by the unweighted reconstruction, but positive
    Kokkos::View<Real*, DeviceMemorySpace> radii2("gnet radii2", n);
    Kokkos::deep_copy(radii2, 1.0);
    SearchBins(queries, grid_, count_, ptr, idx);
    return {ptr, idx, radii2};
  }

protected:
  std::shared_ptr<pcms::BoundPoints> BindImpl(
    const pcms::BindRequest&) const override
  {
    throw pcms::pcms_error("the GNET search cannot be bound; bind on the "
                           "geometry of the density");
  }

private:
  std::shared_ptr<const pcms::CoordinateSystem> system_;
  detail::GnetGrid grid_;
  int count_;
};

// The markers of the file as the DOF holders (the sites) of a layout whose
// geometry is the GNET search.
class GnetSitesLayout final : public pcms::FieldLayout
{
public:
  GnetSitesLayout(std::shared_ptr<const pcms::Geometry> geometry,
                  const std::vector<Real>& sites)
    : n_(static_cast<LO>(sites.size() / Dim))
  {
    SetGeometry(std::move(geometry));
    SetCoordinateSystem(GetGeometry()->GetCoordinateSystem());
    Kokkos::View<Real**, HostMemorySpace> host(
      Kokkos::view_alloc("gnet sites", Kokkos::WithoutInitializing), n_, Dim);
    for (LO i = 0; i < n_; ++i) {
      for (int d = 0; d < Dim; ++d) {
        host(i, d) = sites[static_cast<std::size_t>(i) * Dim + d];
      }
    }
    // the device's own layout
    sites_ = Kokkos::View<Real**, DeviceMemorySpace>("gnet sites", n_, Dim);
    pcms::DeepCopyMismatchLayouts(sites_, host);
    owned_ = std::make_unique<bool[]>(n_);
    std::fill_n(owned_.get(), n_, true);
    gids_.resize(n_);
    ids_.resize(n_);
    for (LO i = 0; i < n_; ++i) {
      gids_[i] = i;
      ids_[i] = i;
    }
    classification_.assign(n_, 0);
  }

  int GetNumComponents() const override { return 1; }
  LO GetNumDofHolder() const override { return n_; }
  pcms::GO GetNumGlobalDofHolder() const override { return n_; }
  bool IsDistributed() const override { return false; }
  pcms::Rank1View<const bool, HostMemorySpace> GetOwnedHost() const override
  {
    return pcms::Rank1View<const bool, HostMemorySpace>(owned_.get(), n_);
  }
  pcms::GlobalIDView<HostMemorySpace> GetGidsHost() const override
  {
    return pcms::GlobalIDView<HostMemorySpace>(gids_.data(), gids_.size());
  }
  pcms::EntOffsetsArray GetEntOffsets() const override
  {
    pcms::EntOffsetsArray offsets{};
    std::fill(offsets.begin(), offsets.end(), static_cast<std::size_t>(n_));
    offsets[0] = 0;
    return offsets;
  }
  pcms::CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates()
    const override
  {
    return pcms::CoordinateView<DeviceMemorySpace>(
      GetCoordinateSystem(), pcms::MakeConstRank2View(sites_));
  }
  int GetDimension() const override { return Dim; }
  // the supports are marker indices: the sites are vertices
  int GetDOFHolderEntityDim() const override { return 0; }
  pcms::Rank1View<const LO, HostMemorySpace>
  GetDOFHolderClassificationDimensionsHost() const override
  {
    return pcms::Rank1View<const LO, HostMemorySpace>(classification_.data(),
                                                      classification_.size());
  }
  pcms::Rank1View<const LO, HostMemorySpace> GetDOFHolderClassificationIdsHost()
    const override
  {
    return pcms::Rank1View<const LO, HostMemorySpace>(ids_.data(), ids_.size());
  }

private:
  LO n_;
  Kokkos::View<Real**, DeviceMemorySpace> sites_;
  std::unique_ptr<bool[]> owned_;
  std::vector<pcms::GO> gids_;
  std::vector<LO> ids_;
  std::vector<LO> classification_;
};

// the number of monomials of degree <= `degree` in 5 variables
std::size_t BasisSize(unsigned degree)
{
  std::size_t size = 1;
  for (unsigned i = 1; i <= Dim; ++i) {
    size = size * (degree + i) / i;
  }
  return size;
}

} // namespace

GnetDistribution OpenGnet(const std::string& path, const GnetOptions& options)
{
  const detail::GnetData data = detail::ReadGnet(path);
  if (options.degree < 1) {
    throw pcms::pcms_error("fusionio::OpenGnet: the degree must be >= 1");
  }
  const detail::GnetGrid grid{static_cast<std::size_t>(data.num_rho),
                              static_cast<std::size_t>(data.num_zeta),
                              static_cast<std::size_t>(data.num_pitch),
                              static_cast<std::size_t>(data.num_energy),
                              data.max_zeta_rad,
                              data.max_energy_ev};
  const int count = detail::NumNeighbourBins(BasisSize(options.degree));
  if (std::pow(count, Dim) > detail::MaxGnetSupports) {
    throw pcms::pcms_error("fusionio::OpenGnet: degree " +
                           std::to_string(options.degree) +
                           " needs more supports than the search provides");
  }
  const auto fewest =
    std::min({grid.num_rho, grid.num_zeta, grid.num_pitch, grid.num_energy});
  if (fewest < static_cast<std::size_t>(count)) {
    throw pcms::pcms_error(
      "fusionio::OpenGnet: " + path + " has fewer than " +
      std::to_string(count) + " bins in a coordinate, which degree " +
      std::to_string(options.degree) + " needs around a point");
  }

  // the bin centres, in marker order
  std::vector<Real> sites;
  sites.reserve(data.NumMarkers() * Dim);
  for (int rho = 0; rho < data.num_rho; ++rho) {
    const int num_theta = detail::GnetData::NumPoloidalBins(rho);
    for (int theta = 0; theta < num_theta; ++theta) {
      for (int zeta = 0; zeta < data.num_zeta; ++zeta) {
        for (int energy = 0; energy < data.num_energy; ++energy) {
          for (int pitch = 0; pitch < data.num_pitch; ++pitch) {
            sites.push_back((0.5 + rho) / data.num_rho);
            sites.push_back((theta + 0.5) * (TwoPi / num_theta));
            sites.push_back((zeta + 0.5) * (data.max_zeta_rad / data.num_zeta));
            sites.push_back(data.max_energy_ev * (energy + 0.5) /
                            data.num_energy);
            sites.push_back(2 * (pitch + 0.5) / data.num_pitch - 1.0);
          }
        }
      }
    }
  }

  // (rho, theta, zeta, energy, pitch)
  std::vector<pcms::Identification> identifications;
  if (options.periodic) {
    identifications = {{.axis = 1, .length = TwoPi},
                       {.axis = 2, .length = data.max_zeta_rad}};
  }
  auto system = pcms::csys::CreateParameterSystem("gnet_phase", Dim,
                                                  std::move(identifications));
  auto layout = std::make_shared<GnetSitesLayout>(
    std::make_shared<GnetGeometry>(system, grid, count), sites);
  auto space = pcms::CreatePolynomialReconstructionSpace(
    pcms::point_cloud::CreateGeometry(std::move(layout)),
    pcms::MLSOptions{.degree = options.degree,
                     .lambda = options.lambda,
                     .basis = pcms::RadialBasisFunction::NO_OP});
  auto density = space->CreateFunction<Real>(
    pcms::FieldMetadata{.name = "density", .extent = pcms::Extent::Intensive});
  const std::vector<double> values = data.NumberDensity();
  density.SetDOFHolderDataHost(pcms::ValueView<const Real, HostMemorySpace>(
    pcms::ValueBasis{}, pcms::Rank2View<const Real, HostMemorySpace>(
                          values.data(), values.size(), 1)));
  const auto [min_volume, max_volume] =
    std::minmax_element(data.volume3d.begin(), data.volume3d.end());
  return {std::move(density), data.max_zeta_rad, data.max_energy_ev,
          *min_volume, *max_volume};
}

} // namespace fusionio
