// fio_trace: field-line tracing and Poincare plots, ported from Fusion-IO
// 1.x's trace tool (trace/main.cpp, trace/trace_integrator.cpp, Copyright (c)
// 2024 Nathaniel Ferraro, MIT License) to the batched fusionio/PCMS API.
//
// The command line, output files (out<n>, q.out, gplot, gplotpsi, gplot3D)
// and the fixed-step RK4 integration in phi are those of trace. What changes
// is the order of work: trace integrates one field line at a time with one
// Fusion-IO call per point, while this integrates all of a rank's field lines
// together, with one batched evaluation per RK4 stage. All lines start at the
// same phi and take the same steps, so they stay in lockstep; a line that
// leaves the domain is dropped.
//
// Not ported: the adaptive RK(5,4) integrator (-adaptive), surfaces of seed
// points (-ds, unimplemented in trace), and the DIII-D I-coil source
// (-diiid-i, disabled in trace).

#include <fusionio/fusionio.hpp>
#include <pcms/field/coordinate_view.hpp>
#include <pcms/field/function_space.h>
#include <pcms/field/value_view.hpp>
#include <pcms/utility/arrays.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace
{

using fusionio::Real;
constexpr Real Pi = std::numbers::pi;

struct TraceSource
{
  std::shared_ptr<fusionio::Source> source;
  pcms::Function<Real> field;
  std::optional<pcms::Function<Real>> psi_norm;
  Real magaxis[2] = {0, 0};
};

struct Parameters
{
  int transits = 100;
  int steps_per_transit = 100;
  int surfaces = 11;
  Real dR = 0, dZ = 0, dR0 = 0, dZ0 = 0;
  Real phi0 = 0;
  Real R0 = 0, Z0 = 0;
  bool R0_set = false, Z0_set = false;
  Real angle = 0;
  bool qout = true;
  bool pout = true;
  bool reverse = false;
  int nplanes = 1;
  bool timing = false;
};

/// Evaluates a function at batches of (R, phi, Z) points. The point and value
/// views are kept for the next batch of the same size, so a tracer's RK4
/// stages allocate them once.
class BatchEvaluator
{
public:
  using Values =
    Kokkos::View<Real**, pcms::DeviceMemorySpace>::host_mirror_type;

  explicit BatchEvaluator(const pcms::Function<Real>& function)
    : function_(&function)
  {
  }

  /// Values at the points in `xyz` (row-major), one row per point; rows of
  /// points outside the domain are NaN. Valid until the next call.
  Values operator()(const std::vector<Real>& xyz)
  {
    const auto& space = function_->GetSpace();
    const size_t n = xyz.size() / 3;
    if (points_.extent(0) != n) {
      const auto num_components =
        static_cast<size_t>(function_->GetLayout().GetNumComponents());
      points_ = Kokkos::View<Real**, pcms::DeviceMemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n, 3);
      values_ = Kokkos::View<Real**, pcms::DeviceMemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "values"), n,
        num_components);
      // the values themselves where device memory is host memory
      result_ = Kokkos::create_mirror_view(values_);
    }
    Kokkos::View<const Real**, Kokkos::LayoutRight, pcms::HostMemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_points(xyz.data(), n, 3);
    pcms::DeepCopyMismatchLayouts(points_, host_points);
    const auto bound = space.Bind(
      pcms::CoordinateView<pcms::DeviceMemorySpace>(
        space.GetCoordinateSystem(), pcms::MakeRank2View(points_)),
      {pcms::OutOfBoundsMode::FILL, std::numeric_limits<Real>::quiet_NaN()});
    const auto evaluator = space.CreatePointEvaluator<Real>(bound);
    evaluator->Evaluate(
      *function_,
      pcms::ValueView<Real, pcms::DeviceMemorySpace>(
        function_->GetData().GetValueBasis(), pcms::MakeRank2View(values_)));
    Kokkos::deep_copy(result_, values_);
    return result_;
  }

private:
  const pcms::Function<Real>* function_;
  Kokkos::View<Real**, pcms::DeviceMemorySpace> points_;
  Kokkos::View<Real**, pcms::DeviceMemorySpace> values_;
  Values result_;
};

/// Field lines integrated together in phi (trace_integrator::integrate with
/// step_rk4, one line per seed point).
class FieldLines
{
public:
  FieldLines(const std::vector<TraceSource>& sources, const Parameters& p,
             Real period)
    : sources_(sources), p_(p), period_(period)
  {
    for (const auto& source : sources_) {
      fields_.emplace_back(source.field);
    }
    if (sources_[0].psi_norm) {
      psi_norm_.emplace(*sources_[0].psi_norm);
    }
  }

  struct Line
  {
    Real R, Z;
    bool active = true;
    // integrator_data
    int toroidal_transits = 0;
    int poloidal_transits = 0;
    Real q = 0;
    Real distance = 0;
    // poloidal transit bookkeeping
    bool add = false;
    Real avg_steps_per_pol_transit = 0;
    Real steps_since_pol_transit = 0;
    std::vector<std::array<Real, 5>> punctures;
  };

  /// Integrates `lines` from phi0; lines that leave the domain become
  /// inactive.
  void Integrate(std::vector<Line>& lines, Real phi0)
  {
    const Real pd = p_.reverse ? -period_ : period_;
    const Real dphi = pd / p_.steps_per_transit;
    const int steps = p_.transits * p_.steps_per_transit + 1;
    const Real R0 = sources_[0].magaxis[0];
    const Real Z0 = sources_[0].magaxis[1];
    Real phi = phi0;
    int k = 0;
    for (int i = 0; i < steps; ++i) {
      Real next_phi = phi + dphi;
      if (p_.reverse ? next_phi <= pd : next_phi >= pd) {
        next_phi -= pd;
        phi -= pd;
      }
      bool plot = false;
      Real pl = p_.angle;
      if (p_.nplanes <= 1) {
        plot = p_.reverse ? (phi > pl && next_phi <= pl)
                          : (phi < pl && next_phi >= pl);
      } else {
        for (int j = 0; j < p_.nplanes; ++j) {
          pl = pd * j / p_.nplanes + p_.angle;
          plot = p_.reverse ? (phi > pl && next_phi <= pl)
                            : (phi < pl && next_phi >= pl);
          if (plot) {
            break;
          }
        }
      }

      std::vector<Real> last_R(lines.size()), last_Z(lines.size());
      for (size_t l = 0; l < lines.size(); ++l) {
        last_R[l] = lines[l].R;
        last_Z[l] = lines[l].Z;
      }
      StepRK4(lines, phi, dphi);
      const Real last_phi = phi;
      phi += dphi;

      k++;
      const bool transit = k == p_.steps_per_transit;
      if (transit) {
        k = 0;
      }
      for (size_t l = 0; l < lines.size(); ++l) {
        auto& line = lines[l];
        if (!line.active) {
          continue;
        }
        const Real R = line.R, Z = line.Z;
        line.distance += std::sqrt((R - last_R[l]) * (R - last_R[l]) +
                                   (Z - last_Z[l]) * (Z - last_Z[l]) +
                                   R * last_R[l] * dphi * dphi);
        if (transit) {
          line.toroidal_transits++;
        }
        const bool ptrans =
          (last_Z[l] < Z0 && Z >= Z0) || (last_Z[l] > Z0 && Z <= Z0);
        if (ptrans) {
          if (line.add) {
            line.poloidal_transits++;
            line.q = static_cast<Real>(line.toroidal_transits) /
                     line.poloidal_transits;
            line.avg_steps_per_pol_transit =
              (line.avg_steps_per_pol_transit * (line.poloidal_transits - 1.) +
               line.steps_since_pol_transit) /
              line.poloidal_transits;
            line.steps_since_pol_transit = 0;
          }
          line.add = !line.add;
        } else {
          line.steps_since_pol_transit++;
        }
      }
      if (plot) {
        SavePunctures(lines, last_R, last_Z, pl, last_phi, dphi, R0, Z0);
      }
    }
    for (auto& line : lines) {
      if (line.active) {
        const Real denom =
          line.poloidal_transits +
          line.steps_since_pol_transit / line.avg_steps_per_pol_transit;
        line.q = line.toroidal_transits / denom;
      }
    }
  }

  double eval_seconds = 0;
  long evaluations = 0;

private:
  /// Sum of the sources' fields at the active lines' points (R + dR, phi,
  /// Z + dZ); lines outside the domain become inactive.
  std::vector<std::array<Real, 3>> Field(std::vector<Line>& lines, Real phi,
                                         const std::vector<Real>& dR,
                                         const std::vector<Real>& dZ)
  {
    std::vector<Real> xyz;
    std::vector<size_t> index;
    for (size_t l = 0; l < lines.size(); ++l) {
      if (lines[l].active) {
        xyz.insert(xyz.end(), {lines[l].R + dR[l], phi, lines[l].Z + dZ[l]});
        index.push_back(l);
      }
    }
    std::vector<std::array<Real, 3>> b(lines.size(), {0, 0, 0});
    if (index.empty()) {
      return b;
    }
    const auto start = std::chrono::steady_clock::now();
    for (auto& field : fields_) {
      const auto values = field(xyz);
      for (size_t i = 0; i < index.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
          b[index[i]][c] += values(i, c);
        }
      }
    }
    eval_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
        .count();
    evaluations += static_cast<long>(index.size());
    for (size_t i : index) {
      if (std::isnan(b[i][0]) || std::isnan(b[i][1]) || std::isnan(b[i][2])) {
        lines[i].active = false;
        std::cerr << "lost at (" << lines[i].R << ", " << lines[i].Z << ")"
                  << std::endl;
      }
    }
    return b;
  }

  // trace_integrator::step_rk4, for every active line. Lost lines are
  // skipped; a line lost in any stage keeps its position from before the
  // step.
  void StepRK4(std::vector<Line>& lines, Real phi, Real dphi)
  {
    const size_t n = lines.size();
    std::vector<Real> k1R(n), k1Z(n), k2R(n), k2Z(n), k3R(n), k3Z(n), k4R(n),
      k4Z(n), dR(n, 0), dZ(n, 0);
    auto b = Field(lines, phi, dR, dZ);
    for (size_t l = 0; l < n; ++l) {
      if (!lines[l].active) {
        continue;
      }
      k1R[l] = lines[l].R * dphi * b[l][0] / b[l][1];
      k1Z[l] = lines[l].R * dphi * b[l][2] / b[l][1];
      dR[l] = k1R[l] / 2.;
      dZ[l] = k1Z[l] / 2.;
    }
    b = Field(lines, phi + dphi / 2., dR, dZ);
    for (size_t l = 0; l < n; ++l) {
      if (!lines[l].active) {
        continue;
      }
      const Real RR = lines[l].R + k1R[l] / 2.;
      k2R[l] = RR * dphi * b[l][0] / b[l][1];
      k2Z[l] = RR * dphi * b[l][2] / b[l][1];
      dR[l] = k2R[l] / 2.;
      dZ[l] = k2Z[l] / 2.;
    }
    b = Field(lines, phi + dphi / 2., dR, dZ);
    for (size_t l = 0; l < n; ++l) {
      if (!lines[l].active) {
        continue;
      }
      const Real RR = lines[l].R + k2R[l] / 2.;
      k3R[l] = RR * dphi * b[l][0] / b[l][1];
      k3Z[l] = RR * dphi * b[l][2] / b[l][1];
      dR[l] = k3R[l];
      dZ[l] = k3Z[l];
    }
    b = Field(lines, phi + dphi, dR, dZ);
    for (size_t l = 0; l < n; ++l) {
      if (!lines[l].active) {
        continue;
      }
      const Real RR = lines[l].R + k3R[l];
      k4R[l] = RR * dphi * b[l][0] / b[l][1];
      k4Z[l] = RR * dphi * b[l][2] / b[l][1];
      lines[l].R += k1R[l] / 6. + k2R[l] / 3. + k3R[l] / 3. + k4R[l] / 6.;
      lines[l].Z += k1Z[l] / 6. + k2Z[l] / 3. + k3Z[l] / 3. + k4Z[l] / 6.;
    }
  }

  // The intercepts of the plotting plane pl crossed in the last step, with
  // theta about the axis and the normalized flux there (evaluated at the
  // first plane, as trace does).
  void SavePunctures(std::vector<Line>& lines, const std::vector<Real>& last_R,
                     const std::vector<Real>& last_Z, Real pl, Real last_phi,
                     Real dphi, Real R0, Real Z0)
  {
    const Real f = (pl - last_phi) / dphi;
    std::vector<Real> xyz;
    std::vector<size_t> index;
    for (size_t l = 0; l < lines.size(); ++l) {
      if (lines[l].active) {
        const Real R = lines[l].R * f + last_R[l] * (1. - f);
        const Real Z = lines[l].Z * f + last_Z[l] * (1. - f);
        xyz.insert(xyz.end(), {R, p_.angle, Z});
        index.push_back(l);
      }
    }
    if (index.empty()) {
      return;
    }
    std::vector<Real> psi(index.size(), 0);
    if (psi_norm_) {
      const auto values = (*psi_norm_)(xyz);
      for (size_t i = 0; i < index.size(); ++i) {
        psi[i] = std::isnan(values(i, 0)) ? 0 : values(i, 0);
      }
    }
    for (size_t i = 0; i < index.size(); ++i) {
      const Real R = xyz[3 * i], Z = xyz[3 * i + 2];
      const Real theta = std::atan2(Z - Z0, R - R0) * 180. / Pi;
      lines[index[i]].punctures.push_back(
        {pl * 180. / Pi, R, Z, theta, psi[i]});
    }
  }

  const std::vector<TraceSource>& sources_;
  const Parameters& p_;
  Real period_;
  // the sources' fields and the first source's normalized flux
  std::vector<BatchEvaluator> fields_;
  std::optional<BatchEvaluator> psi_norm_;
};

// trace's create_source: -m3dc1 <file> <ts> <factor> <shift>, -geqdsk <file>,
// -gpec <dir>.
TraceSource CreateSource(fusionio::Format format,
                         const std::vector<std::string>& a)
{
  if (a.empty() && format != fusionio::Format::M3DC1) {
    throw std::runtime_error("a file name must be provided");
  }
  const std::string path = a.empty() ? "C1.h5" : a[0];
  auto source = fusionio::Open(format, path);
  fusionio::Options options;
  Real slice_time = 0;
  if (format == fusionio::Format::M3DC1) {
    options.part = fusionio::Part::Perturbed;
    options.timeslice = -1;
  }
  if (a.size() >= 2) {
    const int timeslice = std::atoi(a[1].c_str());
    std::cerr << "Time slice " << timeslice << std::endl;
    options.timeslice = timeslice;
    const auto times = source->Times();
    if (timeslice >= 0 && timeslice < static_cast<int>(times.size())) {
      slice_time = times[timeslice];
      std::cerr << "Slice time = " << slice_time << std::endl;
    }
  }
  if (a.size() >= 3) {
    options.linear_scale = std::atof(a[2].c_str());
  }
  if (a.size() >= 4) {
    options.phase = std::atof(a[3].c_str()) * Pi / 180.;
  }
  TraceSource result{
    source, source->CreateFunction(fusionio::Quantity::MagneticField, options)};
  auto equilibrium = options;
  equilibrium.part = fusionio::Part::Equilibrium;
  try {
    result.psi_norm =
      source->CreateFunction(fusionio::Quantity::PoloidalFluxNorm, equilibrium);
  } catch (const std::exception&) {
    std::cerr << "Warning: couldn't open psi_norm field" << std::endl;
  }
  const auto R =
    source->EvaluateSeries(fusionio::Series::MagneticAxisR, slice_time);
  const auto Z =
    source->EvaluateSeries(fusionio::Series::MagneticAxisZ, slice_time);
  if (!R || !Z) {
    throw std::runtime_error("couldn't load the magnetic axis");
  }
  result.magaxis[0] = *R;
  result.magaxis[1] = *Z;
  return result;
}

void PrintHelp()
{
  std::cout
    << "Usage:\n"
    << "\tfio_trace <sources> -dR <dR> -dZ <dZ> -dR0 <dR0> -dZ0 <dZ0> /\n"
    << "\t   -p <pts> -t <trans> -s <steps> -a <angle> -phi0 <phi0> \n"
    << "\t   -pout <pplot> -qout <qplot> -n <nplanes> -timing\n\n"
    << " As Fusion-IO 1.x's trace (see trace -h), with -timing to print the\n"
    << " time spent evaluating the field. Sources: -m3dc1 <c1_file> <ts>\n"
    << " <factor> <shift>, -geqdsk <gfile>, -gpec <dir>.\n"
    << std::endl;
}

bool ParseCommandLine(int argc, char* argv[], Parameters& p,
                      std::vector<TraceSource>& sources)
{
  const std::vector<std::string> options = {
    "-gpec", "-geqdsk", "-m3dc1", "-dR",      "-dZ",   "-dR0",   "-dZ0",
    "-R0",   "-Z0",     "-p",     "-t",       "-s",    "-a",     "-pout",
    "-qout", "-phi0",   "-n",     "-reverse", "-tavg", "-timing"};
  std::vector<std::pair<std::string, std::vector<std::string>>> parsed;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (std::find(options.begin(), options.end(), arg) != options.end()) {
      parsed.push_back({arg, {}});
    } else if (arg == "-adaptive" || arg == "-ds" || arg == "-diiid-i") {
      std::cerr << arg << " is not supported by fio_trace" << std::endl;
      return false;
    } else if (parsed.empty()) {
      std::cerr << "Unrecognized option " << arg << std::endl;
      return false;
    } else {
      parsed.back().second.push_back(arg);
    }
  }
  for (const auto& [opt, a] : parsed) {
    const auto one = [&]() {
      if (a.size() != 1) {
        throw std::runtime_error("Incorrect number of arguments for option " +
                                 opt);
      }
      return a[0];
    };
    if (opt == "-m3dc1") {
      sources.push_back(CreateSource(fusionio::Format::M3DC1, a));
    } else if (opt == "-geqdsk") {
      sources.push_back(CreateSource(fusionio::Format::GEQDSK, a));
    } else if (opt == "-gpec") {
      sources.push_back(CreateSource(fusionio::Format::GPEC, a));
    } else if (opt == "-dR") {
      p.dR = std::atof(one().c_str());
    } else if (opt == "-dZ") {
      p.dZ = std::atof(one().c_str());
    } else if (opt == "-dR0") {
      p.dR0 = std::atof(one().c_str());
    } else if (opt == "-dZ0") {
      p.dZ0 = std::atof(one().c_str());
    } else if (opt == "-R0") {
      p.R0 = std::atof(one().c_str());
      p.R0_set = true;
    } else if (opt == "-Z0") {
      p.Z0 = std::atof(one().c_str());
      p.Z0_set = true;
    } else if (opt == "-p") {
      p.surfaces = std::atoi(one().c_str());
    } else if (opt == "-t") {
      p.transits = std::atoi(one().c_str());
    } else if (opt == "-s") {
      p.steps_per_transit = std::atoi(one().c_str());
    } else if (opt == "-a") {
      p.angle = std::atof(one().c_str()) * Pi / 180.;
    } else if (opt == "-phi0") {
      p.phi0 = std::atof(one().c_str()) * Pi / 180.;
    } else if (opt == "-n") {
      p.nplanes = std::atoi(one().c_str());
    } else if (opt == "-pout") {
      p.pout = std::atoi(one().c_str()) == 1;
    } else if (opt == "-qout") {
      p.qout = std::atoi(one().c_str()) == 1;
    } else if (opt == "-reverse") {
      p.reverse = true;
    } else if (opt == "-tavg") {
      one(); // accepted for compatibility; trace does not use it either
    } else if (opt == "-timing") {
      p.timing = true;
    }
  }
  return true;
}

void WriteGnuplotScripts(const Parameters& p)
{
  std::ofstream plot("gplot");
  plot << "set size ratio -1\nset xlabel 'R'\nset ylabel 'Z'\n"
       << "set pointsize 0.1\nset mapping cartesian\nplot ";
  for (int j = 0; j < p.surfaces; ++j) {
    plot << (j > 0 ? ",\\\n" : "") << "'out" << j << "' using 2:3 title ''";
  }
  plot << std::endl;
  std::ofstream psi("gplotpsi");
  psi << "set size ratio 1\nset xlabel 'theta'\nset ylabel 'psi'\n"
      << "set pointsize 0.1\nset mapping cartesian\nplot [-180:180] ";
  for (int j = 0; j < p.surfaces; ++j) {
    psi << (j > 0 ? ",\\\n" : "") << "'out" << j << "' using 4:5 title ''";
  }
  psi << std::endl;
  std::ofstream plot3d("gplot3D");
  plot3d << "set size ratio 1\nset xlabel 'X'\nset ylabel 'Y'\n"
         << "set zlabel 'Z'\nset pointsize 0.1\nset mapping cylindrical\n"
         << "set angles degrees\nsplot ";
  for (int j = 0; j < p.surfaces; ++j) {
    plot3d << (j > 0 ? ",\\\n" : "") << "'out" << j << "' u 1:3:2"
           << (p.nplanes > 12 ? " w lines" : "") << " title ''";
  }
  plot3d << std::endl;
}

int Run(int argc, char* argv[], int rank, int size)
{
  Parameters p;
  std::vector<TraceSource> sources;
  if (!ParseCommandLine(argc, argv, p, sources)) {
    PrintHelp();
    return 1;
  }
  if (sources.empty()) {
    std::cerr << "No sources specified.  Returning." << std::endl;
    PrintHelp();
    return 1;
  }
  // fusionio sources are cylindrical, which trace calls toroidal.
  std::cerr << "Geometry: toroidal" << std::endl;
  const Real period = sources[0].source->Period();
  std::cerr << "Period: " << period << std::endl;
  const Real R_axis = sources[0].magaxis[0];
  const Real Z_axis = sources[0].magaxis[1];
  std::cerr << "Magnetic axis is at ( " << R_axis << ", " << Z_axis << ")"
            << std::endl;
  if (!p.R0_set) {
    p.R0 = R_axis;
  }
  if (!p.Z0_set) {
    p.Z0 = Z_axis;
  }
  if (p.dR0 == 0.) {
    p.dR0 = p.dR;
  }
  if (rank == 0 && p.pout) {
    WriteGnuplotScripts(p);
  }

  // Surfaces are dealt round-robin to ranks, as in trace.
  std::vector<int> mine;
  for (int s = rank; s < p.surfaces; s += size) {
    mine.push_back(s);
  }
  std::vector<FieldLines::Line> lines;
  for (const int s : mine) {
    lines.push_back({p.R0 + p.dR0 + p.dR * s, p.Z0 + p.dZ0 + p.dZ * s});
  }
  FieldLines tracer(sources, p, period);
  const auto start = std::chrono::steady_clock::now();
  tracer.Integrate(lines, p.phi0);
  const double seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
  std::cerr << "Computation completed in " << seconds << " s." << std::endl;
  if (p.timing) {
    std::cerr << "rank " << rank << ": " << lines.size() << " field lines, "
              << tracer.evaluations << " point evaluations, "
              << tracer.eval_seconds << " s evaluating the field ("
              << 1e9 * tracer.eval_seconds / std::max(1L, tracer.evaluations)
              << " ns per point), " << seconds << " s integrating" << std::endl;
  }

  for (size_t j = 0; j < lines.size(); ++j) {
    if (p.pout) {
      std::ofstream out("out" + std::to_string(mine[j]));
      for (const auto& row : lines[j].punctures) {
        for (const Real v : row) {
          out << std::setiosflags(std::ios::scientific) << std::setw(20)
              << std::setprecision(12) << v;
        }
        out << '\n';
      }
    }
  }

  // Gather r and q of every surface on rank 0, in surface order.
  // q.out holds the mean, min and max of q over a surface's seed points,
  // trace's mean weighted by toroidal transits; with one seed point per
  // surface they agree, except that the mean is NaN without a full transit.
  std::vector<Real> local(3 * lines.size());
  for (size_t j = 0; j < lines.size(); ++j) {
    const int s = mine[j];
    const Real R = p.R0 + p.dR0 + p.dR * s;
    const Real Z = p.Z0 + p.dZ0 + p.dZ * s;
    const Real transits = lines[j].toroidal_transits;
    local[3 * j] =
      std::sqrt((R - R_axis) * (R - R_axis) + (Z - Z_axis) * (Z - Z_axis));
    // trace's transit-weighted mean, kept as q * transits / transits so it
    // matches trace bit for bit (and is NaN without a full transit).
    local[3 * j + 1] = lines[j].q * transits / transits;
    local[3 * j + 2] = lines[j].q;
  }
  std::vector<int> counts(size), offsets(size);
  const int count = static_cast<int>(local.size());
  MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  for (int r = 1; r < size; ++r) {
    offsets[r] = offsets[r - 1] + counts[r - 1];
  }
  std::vector<Real> all(3 * p.surfaces);
  MPI_Gatherv(local.data(), count, MPI_DOUBLE, all.data(), counts.data(),
              offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank == 0 && p.qout) {
    std::ofstream q("q.out");
    for (int s = 0; s < p.surfaces; ++s) {
      // Surface s is the (s / size)-th surface of rank s % size.
      const int i = offsets[s % size] + 3 * (s / size);
      q << all[i] << '\t' << all[i + 1] << '\t' << all[i + 2] << '\t'
        << all[i + 2] << '\n';
    }
  }
  if (rank == 0) {
    std::cerr << "===============================\n"
              << "Poincare computation complete.\n"
              << "To view Poincare plot, use 'gnuplot gplot'" << std::endl;
  }
  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      result = Run(argc, argv, rank, size);
    } catch (const std::exception& e) {
      std::cerr << "fio_trace: " << e.what() << std::endl;
      result = 1;
    }
  }
  MPI_Finalize();
  return result;
}
