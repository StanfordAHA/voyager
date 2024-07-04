#pragma once

#include "test/common/operations/Common.h"

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T>
inline ACCUMULATE_T *matrix_vector_multiply(const INPUT_T *inputs,
                                            const INPUT_T *weights,
                                            const INPUT_T *bias,
                                            const codegen::MatrixParam &param) {
  const auto weight = param.weight();
  int K = weight.shape(0);
  int C = weight.shape(1);

  bool input_double_precision = is_double_precision(param.input());
  bool weight_double_precision = is_double_precision(param.weight());
  bool bias_double_precision = is_double_precision(param.bias());

  ACCUMULATE_T *outputs = new ACCUMULATE_T[K];
  for (int i = 0; i < K; i++) {
    outputs[i] = 0.0;
  }

  for (int k = 0; k < K; k++) {
    ACCUMULATE_T product[C];
    for (int c = 0; c < C; c++) {
      ACCUMULATE_T input = read_tensor(inputs, c, input_double_precision);
      ACCUMULATE_T weight =
          read_tensor(weights, k * C + c, weight_double_precision);
      product[c] = static_cast<ACCUMULATE_T>(input * weight);
    }

    // perform a tree addition
    for (int i = 0; i < C; i += OC_DIMENSION) {
      ACCUMULATE_T buffer[OC_DIMENSION];
      for (int j = 0; j < OC_DIMENSION; j++) {
        buffer[j] = product[i + j];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int j = 0; j < depth; j += 2) {
          buffer[j / 2] = static_cast<ACCUMULATE_T>(buffer[j] + buffer[j + 1]);
        }
        depth = depth / 2;
      }
      outputs[k] = static_cast<ACCUMULATE_T>(outputs[k] + buffer[0]);
    }

    if (param.has_bias()) {
      outputs[k] += read_tensor(bias, k, true);
    }
  }
  return outputs;
}
