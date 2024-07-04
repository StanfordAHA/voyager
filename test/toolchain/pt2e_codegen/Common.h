#pragma once

#include <map>
#include <set>

#include "src/Params.h"
#include "test/common/VerificationTypes.h"

std::map<int, std::set<std::string>> vector_ops = {
    {0, {"add", "add_", "sub", "sub_", "mul", "mul_"}},
    {1, {"exp"}},
    {2, {}},
    {3, {"add", "add_", "mul", "mul_", "square"}},
    {4, {"relu", "relu_"}},
};

std::map<std::string, unsigned int> get_vector_instruction_mapping() {
  std::map<std::string, unsigned int> mapping;
  mapping["add"] = VectorInstructions::vadd;
  mapping["add_"] = VectorInstructions::vadd;
  mapping["sub"] = VectorInstructions::vsub;
  mapping["sub_"] = VectorInstructions::vsub;
  mapping["mul"] = VectorInstructions::vmult;
  mapping["mul_"] = VectorInstructions::vmult;
  mapping["relu"] = VectorInstructions::vrelu;
  mapping["relu_"] = VectorInstructions::vrelu;
  mapping["exp"] = VectorInstructions::vexp;
  mapping["square"] = VectorInstructions::vsquare;
  return mapping;
}

inline MemorySource get_partition(const int &partition) {
  return partition == 0 ? SRAM : RRAM;
}
