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

void MapOperation(const codegen::Operator &param,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  if (param.has_matrix_op()) {
    const auto matrix_op = param.matrix_op();
    if (matrix_op.opcode() == "layer_norm") {
      MapLayerNorm(param, mappedParams, opMemoryMaps);
      return;
    }

    const auto &inputs = matrix_op.has_mx_input() ? matrix_op.mx_input().input()
                                                  : matrix_op.input();
    int dim = 1;
    for (int i = 0; i < inputs.shape_size() - 1; i++) {
      dim *= inputs.shape(i);
    }

    if (dim == 1) {
      MapMatrixVectorMultiply(param, mappedParams, opMemoryMaps);
    } else {
      MapMatrixOperation(param, mappedParams, opMemoryMaps);
    }
  } else if (param.has_reduce_op()) {
    const auto &reduce_op = param.reduce_op();
    if (reduce_op.opcode() == "softmax") {
      MapSoftmax(param, mappedParams, opMemoryMaps);
    } else if (reduce_op.opcode() == "calculate_mx_qparam") {
      MapMXQparam(param, mappedParams, opMemoryMaps);
    } else {
      std::cerr << "Unsupported reduce instruction: " << reduce_op.opcode()
                << std::endl;
      exit(1);
    }
  } else if (param.has_pooling_op()) {
    MapPoolingOperation(param, mappedParams, opMemoryMaps);
  } else if (param.has_slicing_op() || param.has_reshape_op() ||
             param.vector_ops_size() > 0) {
    MapVectorOperations(param, mappedParams, opMemoryMaps);
  }
}
