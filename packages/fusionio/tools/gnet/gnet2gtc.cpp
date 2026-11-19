// gnet2gtc: map a GNET 5D distribution to GTC markers.
//
// Reads a GNET HDF5 file, reconstructs the distribution with moving least
// squares at random candidate points, keeps the candidates above a threshold
// (rejection sampling) and writes the GTC markers to an HDF5 file. After
// gnet2gtc of the PCMS branch gnet-gtc-coupling.

#include <fusionio/gnet.hpp>
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

// Settings of a GNET -> GTC run (defaults match the reference setup).
struct Options
{
  std::string input;
  std::string output;
  std::size_t samples = 30000000;
  int batch = 1000000;
  std::uint64_t seed = 14000;
  fusionio::GnetOptions gnet;
  double threshold_scale = 1e10; // threshold = threshold_scale / max(volume3d)
  bool stream = false;
};

void PrintUsage(const char* prog)
{
  const Options d;
  std::cout
    << "Usage: " << prog << " --input <gnet.h5> --output <gtc.h5> [options]\n"
    << "\n"
    << "Required:\n"
    << "  --input <file>            GNET HDF5 file\n"
    << "  --output <file>           GTC output HDF5 file (overwritten)\n"
    << "\n"
    << "Options:\n"
    << "  --samples <n>             GTC markers to produce (default "
    << d.samples << ")\n"
    << "  --batch <n>               candidates per sampling round (default "
    << d.batch << ")\n"
    << "  --seed <n>                random seed (default " << d.seed << ")\n"
    << "  --degree <n>              MLS polynomial degree (default "
    << d.gnet.degree << ")\n"
    << "  --lambda <x>              MLS regularization (default "
    << d.gnet.lambda << ")\n"
    << "  --threshold-scale <x>     threshold = scale / max(volume3d) (default "
    << d.threshold_scale << ")\n"
    << "  --no-periodic             fit theta and zeta at their raw "
       "coordinates (the PCMS gnet-gtc-coupling behaviour); by default they "
       "are periodic\n"
    << "  --stream                  write each round's markers to the file as "
       "they are accepted, so memory does not grow with --samples\n"
    << "  --help                    show this message\n"
    << "\n"
    << "Kokkos options (e.g. --kokkos-device-id=<n>) are also accepted.\n";
}

Options ParseArgs(int argc, char* argv[])
{
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + arg);
      }
      return argv[++i];
    };
    try {
      if (arg == "--input") {
        o.input = value();
      } else if (arg == "--output") {
        o.output = value();
      } else if (arg == "--samples") {
        o.samples = std::stoull(value());
      } else if (arg == "--batch") {
        o.batch = std::stoi(value());
      } else if (arg == "--seed") {
        o.seed = std::stoull(value());
      } else if (arg == "--degree") {
        o.gnet.degree = static_cast<unsigned>(std::stoul(value()));
      } else if (arg == "--lambda") {
        o.gnet.lambda = std::stod(value());
      } else if (arg == "--threshold-scale") {
        o.threshold_scale = std::stod(value());
      } else if (arg == "--no-periodic") {
        o.gnet.periodic = false;
      } else if (arg == "--stream") {
        o.stream = true;
      } else {
        throw std::runtime_error("unknown option " + arg);
      }
    } catch (const std::logic_error&) { // std::stoi/stod failures
      throw std::runtime_error("invalid value for " + arg);
    }
  }
  if (o.input.empty() || o.output.empty()) {
    throw std::runtime_error("--input and --output are required");
  }
  if (o.samples == 0 || o.samples > 400000000) {
    throw std::runtime_error("--samples must be in (0, 400000000]");
  }
  if (o.batch <= 0 || o.batch > 400000000) {
    throw std::runtime_error("--batch must be in (0, 400000000]");
  }
  if (o.gnet.degree < 1) {
    throw std::runtime_error("--degree must be >= 1");
  }
  return o;
}

void Run(const Options& o)
{
  std::cout << "GNET -> GTC\n"
            << "  input           : " << o.input << "\n"
            << "  output          : " << o.output << "\n"
            << "  samples         : " << o.samples << "\n"
            << "  batch size      : " << o.batch << "\n"
            << "  seed            : " << o.seed << "\n"
            << "  MLS degree      : " << o.gnet.degree << "\n"
            << "  MLS lambda      : " << o.gnet.lambda << "\n"
            << "  threshold scale : " << o.threshold_scale << "\n"
            << "  periodic        : " << (o.gnet.periodic ? "yes" : "no")
            << "\n";
  Kokkos::Timer timer;
  const auto gnet = fusionio::OpenGnet(o.input, o.gnet);
  const double threshold = o.threshold_scale / gnet.max_volume;
  std::cout << "  [load]       " << timer.seconds() << " s\n"
            << "  volume3d min " << gnet.min_volume << ", max "
            << gnet.max_volume << " -> rejection threshold " << threshold
            << "\n";

  timer.reset();
  const fusionio::SamplingOptions sampling{o.samples, o.batch, o.seed,
                                           threshold};
  const auto domain = fusionio::GTCDomain();
  std::size_t candidates = 0;
  if (o.stream) {
    fusionio::GTCMarkerWriter writer(o.output, o.samples);
    candidates = fusionio::SampleMarkers(
      gnet.density, domain, sampling,
      [&](const fusionio::MarkerRows& rows) { writer.Append(rows); });
    Kokkos::fence();
    std::cout << "  [sampling + write] " << timer.seconds() << " s -> "
              << o.output << "\n";
  } else {
    const auto markers =
      fusionio::SampleMarkers(gnet.density, domain, sampling);
    candidates = markers.candidates;
    Kokkos::fence();
    std::cout << "  [sampling]   " << timer.seconds() << " s\n";
    timer.reset();
    fusionio::WriteGTCMarkers(o.output, markers.rows);
    std::cout << "  [write]      " << timer.seconds() << " s -> " << o.output
              << "\n";
  }
  std::cout << "  accepted " << o.samples << " of " << candidates
            << " candidates\n"
            << "GNET to GTC Mapping Completed Successfully!\n";
}

} // namespace

int main(int argc, char* argv[])
{
  // handle --help before Kokkos, which prints its own help for it
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return EXIT_SUCCESS;
    }
  }
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    Run(ParseArgs(argc, argv));
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
