// make_synthetic_gnet <file> [num_rho num_zeta num_energy num_pitch]: writes
// the synthetic GNET file of the tests.
#include "gnet_synthetic.hpp"
#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc != 2 && argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <file> [num_rho num_zeta num_energy num_pitch]\n";
    return EXIT_FAILURE;
  }
  fusionio::test::GnetShape shape;
  if (argc == 6) {
    shape = {std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
             std::atoi(argv[5])};
  }
  try {
    fusionio::test::WriteSyntheticGnet(argv[1], shape);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
