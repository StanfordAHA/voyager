#pragma once
#define NO_SYSC
#include <vector>
// clang-format off
#include "src/DataTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/UniversalPosit.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

// args contains an array of pointers to the input tensors which serves as the
// positional arguments to the model

void run_pytorch_model(const example::AcceleratorParam &params,
                       std::vector<float *> args);

void run_pytorch_model(const example::AcceleratorParam &params,
                       std::vector<INPUT_DATATYPE *> args);

#ifndef NO_UNIVERSAL
void run_pytorch_model(const example::AcceleratorParam &params,
                       std::vector<UniversalPosit *> args);
#endif
