#pragma once

#include "src/AccelTypes.h"
#include "src/ArchitectureParams.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"

void MapOperation(const SimplifiedParams &params, MatrixParams &matrixParams,
                  bool &matrixParamsValid, VectorParams &vectorParams,
                  VectorInstructionConfig &vectorInstructionConfig,
                  bool &vectorParamsValid);
