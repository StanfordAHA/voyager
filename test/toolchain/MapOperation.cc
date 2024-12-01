#include "src/AccelTypes.h"
#include "src/Params.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"
#include "test/toolchain/LayerNorm.h"
#include "test/toolchain/MatrixOps.h"
#include "test/toolchain/MatrixVectorMultiply.h"
#include "test/toolchain/Microscaling.h"
#include "test/toolchain/Pooling.h"
#include "test/toolchain/Softmax.h"
#include "test/toolchain/VectorOps.h"

void MapOperation(const codegen::AcceleratorParam &param,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  if (param.has_matrix_param()) {
    const auto matrix_param = param.matrix_param();
    if (matrix_param.opcode() == "layer_norm") {
      MapLayerNorm(param, mappedParams, opMemoryMaps);
      return;
    }

    const auto &inputs = matrix_param.has_mx_input()
                             ? matrix_param.mx_input().input()
                             : matrix_param.input();
    int dim = 1;
    for (int i = 0; i < inputs.shape_size() - 1; i++) {
      dim *= inputs.shape(i);
    }

    if (dim == 1) {
      MapMatrixVectorMultiply(param, mappedParams, opMemoryMaps);
    } else {
      MapMatrixOperation(param, mappedParams, opMemoryMaps);
    }
  } else if (param.has_reduce_param()) {
    const auto &reduce_param = param.reduce_param();
    if (reduce_param.opcode() == "softmax") {
      MapSoftmax(param, mappedParams, opMemoryMaps);
    } else if (reduce_param.opcode() == "calculate_mx_qparam") {
      MapMXQparam(param, mappedParams, opMemoryMaps);
    } else {
      std::cerr << "Unsupported reduce instruction: " << reduce_param.opcode()
                << std::endl;
      exit(1);
    }
  } else if (param.has_pooling_param()) {
    MapPoolingOperation(param, mappedParams, opMemoryMaps);
  } else if (param.has_slicing_param() || param.has_reshape_param() ||
             param.vector_params_size() > 0) {
    MapVectorOperations(param, mappedParams, opMemoryMaps);
  }
}
