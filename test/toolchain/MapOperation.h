#pragma once

#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/compiler/proto/param.pb.h"

void MapOperation(const codegen::Operator &param,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps);