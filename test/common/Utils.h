#pragma once

#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "VerificationTypes.h"
#include "src/ArchitectureParams.h"

float compare_arrays(INPUT_DATATYPE *matrixA, std::string matrixA_name,
                     INPUT_DATATYPE *matrixB, std::string matrixB_name,
                     size_t size, std::string filename, bool accType);
float compare_arrays(INPUT_DATATYPE *matrixA, std::string matrixA_name,
                     float *matrixB, std::string matrixB_name, size_t size,
                     std::string filename, bool accType);
float compare_arrays(float *matrixA, std::string matrixA_name, float *matrixB,
                     std::string matrixB_name, size_t size,
                     std::string filename, bool accType);

int validateMapping(Tiling tiling);
