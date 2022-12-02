// This file is the main entry point to running event-level SystemC simulations.
// It can run any model, such as ResNet and mobileBert-Tiny
// Parameters are passed via environment variables, not arguments.
#include <iostream>
#include <string>
#include <vector>

#include "test/common/Simulation.h"

extern "C" int sc_main(int argc, char* argv[]) {
  Simulation sim;
#ifndef SOC_COSIM
  if (argc > 1) {
    std::cerr
        << "ERROR: Don't supply command line arguments, instead use env vars."
        << std::endl;
    sim.print_help();
    std::abort();
  }
#endif
  sim.loadMemory();
  sim.run();
  return sim.checkOutput();
}