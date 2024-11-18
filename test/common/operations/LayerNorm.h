#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T *layer_norm(std::any input_tensor, std::any weight_tensor,
                     std::any bias_tensor, const codegen::MatrixParam &param) {
  T *inputs = std::any_cast<T *>(input_tensor);
  T *weights = param.has_weight() ? std::any_cast<T *>(weight_tensor) : nullptr;
  T *bias = param.has_bias() ? std::any_cast<T *>(bias_tensor) : nullptr;
  T *output = new T[get_size(param.input())];

  const auto &input = param.input();
  const auto input_shape = get_shape(input);
  const int outer_dim = input_shape.back();
  const int inner_dim = get_size(input_shape) / outer_dim;

  T size_inv(1.0 / outer_dim);

  for (int i = 0; i < inner_dim; i++) {
    // In the first pass, scale inputs by 1 / outer_dim
    T normalized_inputs[outer_dim];
    for (int j = 0; j < outer_dim; j++) {
      T normalized_input = inputs[i * outer_dim + j] * size_inv;
      normalized_inputs[j] = normalized_input;
    }

    // Perform a tree addition to compute the mean
    T mean = 0.0;
    for (int j = 0; j < outer_dim; j += OC_DIMENSION) {
      T buffer[OC_DIMENSION];
      for (int k = 0; k < OC_DIMENSION; k++) {
        buffer[k] = normalized_inputs[j + k];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int k = 0; k < depth; k += 2) {
          buffer[k / 2] = static_cast<T>(buffer[k] + buffer[k + 1]);
        }
        depth = depth / 2;
      }
      mean += buffer[0];
    }

    // In the second pass, subtract the mean from the tensor and square the
    // result
    T squares[outer_dim];
    for (int j = 0; j < outer_dim; j++) {
      T input = inputs[i * outer_dim + j] - mean;
      squares[j] = static_cast<T>(input * input);
    }

    // Perform a tree addition to compute the variance
    T variance = 0.0;
    for (int j = 0; j < outer_dim; j += OC_DIMENSION) {
      T buffer[OC_DIMENSION];
      for (int k = 0; k < OC_DIMENSION; k++) {
        buffer[k] = squares[j + k];
      }

      int depth = OC_DIMENSION;
      while (depth > 1) {
        for (int k = 0; k < depth; k += 2) {
          buffer[k / 2] = static_cast<T>(buffer[k] + buffer[k + 1]);
        }
        depth = depth / 2;
      }
      variance += buffer[0];
    }

    // We skipped dividing the variance by outer_dim and adding eps to it
    // T norm_dim(outer_dim);
    // T eps(1e-05);

    // variance /= norm_dim;
    // variance += eps;

    T stddev_inv = variance.inv_sqrt();
    T divisor = sqrt(outer_dim);

    // Normalize by variance and perform an affine transformation
    for (int j = 0; j < outer_dim; j++) {
      T input = inputs[i * outer_dim + j];
      input -= mean;
      input *= stddev_inv;
      input *= divisor;

      // perform affine transformation
      if (weights) input *= weights[j];
      if (bias) input += bias[j];

      output[i * outer_dim + j] = input;
    }
  }

  return output;
}
