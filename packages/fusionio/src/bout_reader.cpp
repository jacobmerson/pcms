#include "fusionio/bout.hpp"
#include "bout_points.hpp"
#include <pcms/field/function_space.hpp>
#include <pcms/field/value_view.hpp>
#include <pcms/utility/assert.hpp>
#include <Kokkos_Core.hpp>
#include <netcdf.h>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <utility>

namespace fusionio
{

namespace
{

using pcms::HostMemorySpace;
using pcms::pcms_error;

class NetcdfFile
{
public:
  explicit NetcdfFile(const std::string& path) : path_(path)
  {
    if (nc_open(path.c_str(), NC_NOWRITE, &id_) != NC_NOERR) {
      throw pcms_error("fusionio::ReadBoutGrid: could not open '" + path + "'");
    }
  }
  ~NetcdfFile() { nc_close(id_); }
  NetcdfFile(const NetcdfFile&) = delete;
  NetcdfFile& operator=(const NetcdfFile&) = delete;

  [[nodiscard]] std::optional<int> Variable(const std::string& name) const
  {
    int var = 0;
    if (nc_inq_varid(id_, name.c_str(), &var) != NC_NOERR) {
      return std::nullopt;
    }
    return var;
  }

  /// the variable's values, of `size` values
  [[nodiscard]] std::vector<Real> Read(const std::string& name,
                                       std::size_t size) const
  {
    const auto var = Variable(name);
    std::size_t found = 0;
    int ndims = 0;
    int dims[NC_MAX_VAR_DIMS];
    if (var && nc_inq_varndims(id_, *var, &ndims) == NC_NOERR &&
        nc_inq_vardimid(id_, *var, dims) == NC_NOERR) {
      found = 1;
      for (int d = 0; d < ndims; ++d) {
        std::size_t length = 0;
        nc_inq_dimlen(id_, dims[d], &length);
        found *= length;
      }
    }
    std::vector<Real> values(size);
    if (found != size ||
        nc_get_var_double(id_, *var, values.data()) != NC_NOERR) {
      throw pcms_error("fusionio::ReadBoutGrid: '" + path_ +
                       "' has no variable " + name + " of " +
                       std::to_string(size) + " values");
    }
    return values;
  }

  [[nodiscard]] int ReadInt(const std::string& name) const
  {
    return static_cast<int>(std::lround(Read(name, 1)[0]));
  }

  [[nodiscard]] int ReadInt(const std::string& name, int fallback) const
  {
    return Variable(name) ? ReadInt(name) : fallback;
  }

private:
  std::string path_;
  int id_ = -1;
};

// the source's quantities: the species and the file's variable
struct BoutQuantity
{
  Quantity quantity;
  std::optional<Species> species;
  const char* variable;
};

constexpr BoutQuantity BoutQuantities[] = {
  {Quantity::TotalPressure, std::nullopt, "pressure"},
  {Quantity::Density, Species::MainIon, "Ni0"},
  {Quantity::Temperature, Species::MainIon, "Ti0"},
  {Quantity::Temperature, Species::Electron, "Te0"},
};

class BoutSource final : public Source
{
public:
  BoutSource(BoutGridFile file, BoutSpaces spaces)
    : file_(std::move(file)), spaces_(std::move(spaces))
  {
  }

  [[nodiscard]] std::shared_ptr<const pcms::Geometry> GetGeometry()
    const override
  {
    return spaces_.geometry;
  }

  [[nodiscard]] std::vector<QuantityInfo> Quantities() const override
  {
    std::vector<QuantityInfo> quantities;
    for (const auto& q : BoutQuantities) {
      const bool listed = std::ranges::any_of(
        quantities, [&](const auto& i) { return i.quantity == q.quantity; });
      if (!listed && file_.arrays.contains(q.variable)) {
        quantities.push_back({q.quantity, 1});
      }
    }
    return quantities;
  }

  [[nodiscard]] std::vector<Real> Times() const override { return {}; }

  [[nodiscard]] std::optional<Box> BoundingBox() const override
  {
    return std::nullopt;
  }

  [[nodiscard]] std::optional<Real> EvaluateSeries(Series /*series*/,
                                                   Real /*time*/) const override
  {
    return std::nullopt;
  }

  [[nodiscard]] pcms::Function<Real> CreateFunction(
    Quantity quantity, const Options& options) const override
  {
    if (options.timeslice || options.part || options.linear_scale ||
        options.phase) {
      throw pcms_error(
        "fusionio: BOUT++ grid sources support only the species option");
    }
    const auto* entry = std::ranges::find_if(BoutQuantities, [&](
                                                               const auto& q) {
      return q.quantity == quantity &&
             (!q.species || q.species == options.species.value_or(*q.species));
    });
    if (entry == std::end(BoutQuantities) ||
        (options.species && !entry->species) ||
        !file_.arrays.contains(entry->variable)) {
      throw pcms_error("fusionio: this BOUT++ grid does not provide " +
                       std::string(Name(quantity)) + " for these options");
    }
    auto basis = QuantityBasis(quantity, nullptr);
    if (quantity == Quantity::Density) {
      // Ni0 is in the file's units (1e20 m^-3 in BOUT++'s tokamak grids)
      basis.units.reset();
    }
    auto function =
      spaces_.space->CreateFunction<Real>(Metadata(quantity), std::move(basis));
    const auto& values = file_.arrays.at(entry->variable);
    const auto points = detail::BlockPoints(*spaces_.space, file_.grid);
    Kokkos::View<Real**, HostMemorySpace> dofs("bout_quantity", points.size(),
                                               1);
    for (std::size_t k = 0; k < points.size(); ++k) {
      // constant in z, so the twist-shifted y = ny plane is the y = 0 plane
      dofs(k, 0) = values[points[k].ix * file_.grid.ny + points[k].jy];
    }
    function.SetDOFHolderDataHost(pcms::ValueView<const Real, HostMemorySpace>(
      function.GetData().GetValueBasis(), pcms::MakeConstRank2View(dofs)));
    return function;
  }

private:
  BoutGridFile file_;
  BoutSpaces spaces_;
};

} // namespace

BoutGridFile ReadBoutGrid(const std::string& path, int nz, int zperiod)
{
  const NetcdfFile nc(path);
  BoutGridFile file;
  auto& grid = file.grid;
  grid.nx = nc.ReadInt("nx");
  grid.ny = nc.ReadInt("ny");
  grid.nz = nz;
  grid.zperiod = zperiod;
  grid.dz = 2 * std::numbers::pi_v<Real> / (zperiod * nz);
  // BOUT++'s defaults: x guard cells in the file, separatrices outside it
  grid.mxg = nc.ReadInt("MXG", 2);
  grid.ixseps1 = nc.ReadInt("ixseps1", grid.nx);
  grid.ixseps2 = nc.ReadInt("ixseps2", grid.nx);
  grid.jyseps1_1 = nc.ReadInt("jyseps1_1", -1);
  grid.jyseps1_2 = nc.ReadInt("jyseps1_2", grid.ny / 2);
  grid.jyseps2_1 = nc.ReadInt("jyseps2_1", grid.ny / 2);
  grid.jyseps2_2 = nc.ReadInt("jyseps2_2", grid.ny - 1);
  const auto size = static_cast<std::size_t>(grid.nx) * grid.ny;
  for (const char* name : {"Rxy", "Zxy", "zShift", "dx", "dy", "Bpxy", "Btxy",
                           "Bxy", "hthe", "sinty"}) {
    file.arrays[name] = nc.Read(name, size);
  }
  file.arrays["ShiftAngle"] = nc.Read("ShiftAngle", grid.nx);
  for (const auto& q : BoutQuantities) {
    if (nc.Variable(q.variable)) {
      file.arrays[q.variable] = nc.Read(q.variable, size);
    }
  }
  // BOUT++'s field-aligned tokamak coordinates (x = psi, y = theta,
  // z = phi - zShift) with the integrated shear I = sinty
  auto& a = file.arrays;
  for (const char* name : {"g11", "g22", "g33", "g12", "g13", "g23"}) {
    a[name].resize(size);
  }
  for (std::size_t p = 0; p < size; ++p) {
    const Real rbp = a["Rxy"][p] * a["Bpxy"][p];
    const Real shear = a["sinty"][p];
    a["g11"][p] = rbp * rbp;
    a["g22"][p] = 1 / (a["hthe"][p] * a["hthe"][p]);
    a["g33"][p] =
      shear * shear * rbp * rbp + a["Bxy"][p] * a["Bxy"][p] / (rbp * rbp);
    a["g12"][p] = 0;
    a["g13"][p] = -shear * rbp * rbp;
    a["g23"][p] = -a["Btxy"][p] / (a["hthe"][p] * rbp);
  }
  const auto view = [&](const char* name) {
    return HostArray2D(a.at(name).data(), grid.nx, grid.ny);
  };
  grid.rxy = view("Rxy");
  grid.zxy = view("Zxy");
  grid.zshift = view("zShift");
  grid.dx = view("dx");
  grid.dy = view("dy");
  grid.shift_angle = HostArray1D(a.at("ShiftAngle").data(), grid.nx);
  grid.g_contra = {view("g11"), view("g22"), view("g33"),
                   view("g12"), view("g13"), view("g23")};
  return file;
}

std::shared_ptr<Source> OpenBout(const std::string& path, int nz, int zperiod)
{
  auto file = ReadBoutGrid(path, nz, zperiod);
  auto spaces = Build(file.grid);
  return std::make_shared<BoutSource>(std::move(file), std::move(spaces));
}

} // namespace fusionio
