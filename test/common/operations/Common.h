#pragma once
#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

using INTERMEDIATE_DTYPE = ACCUM_DATATYPE::AccumulationDatatype;

inline std::vector<int> get_shape(const codegen::Tensor &tensor) {
  const auto repeated_field = tensor.shape();
  std::vector<int> shape(repeated_field.begin(), repeated_field.end());

  if (tensor.has_reshape()) {
    const auto &param = tensor.reshape();
    shape = {param.output_sizes().begin(), param.output_sizes().end()};
  } else if (tensor.has_slicing()) {
    const auto &param = tensor.slicing();
    const int dim = param.dim();
    const int start = param.start();
    const int end = param.end();
    const int step = param.step();
    const int num_elements = (end - start) / step;
    shape[dim] = num_elements;
  }

  return shape;
}

inline int get_size(const std::vector<int> &shape) {
  int size = 1;
  for (const auto &dim : shape) size *= dim;
  return size;
}

inline int get_size(const codegen::Tensor &tensor) {
  const auto shape = get_shape(tensor);
  return get_size(shape);
}

// Helper function to compute strides
inline std::vector<int> compute_strides(const std::vector<int> &shape) {
  std::vector<int> strides(shape.size());
  strides.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Helper function to compute the linear index from multi-dimensional indices
inline int compute_linear_index(const std::vector<int> &indices,
                                const std::vector<int> &strides) {
  int linear_index = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    linear_index += indices[i] * strides[i];
  }
  return linear_index;
}
