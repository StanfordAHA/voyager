#pragma once

#include "test/common/operations/Common.h"

// Helper function to compute strides
inline std::vector<int> compute_strides(const std::vector<int>& shape) {
  std::vector<int> strides(shape.size());
  strides.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Helper function to compute the linear index from multi-dimensional indices
inline int compute_linear_index(const std::vector<int>& indices,
                                const std::vector<int>& strides) {
  int linear_index = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    linear_index += indices[i] * strides[i];
  }
  return linear_index;
}

template <typename T>
inline T* permute(const T* inputs, const codegen::ReshapeParam param) {
  const auto input = param.input();
  const int size = get_size(input);
  T* outputs = new T[size];

  const auto input_shape = get_shape(input);
  std::vector<int> order;
  if (param.opcode() == "permute") {
    order.insert(order.end(), param.dims().begin(), param.dims().end());
  } else if (param.opcode() == "transpose") {
    int dim1 = param.dims(0);
    int dim2 = param.dims(1);

    int size = input.shape_size();
    if (dim1 < 0) dim1 += size;
    if (dim2 < 0) dim2 += size;

    order.insert(order.end(), {0, 1, 2, 3});
    std::swap(order[dim1], order[dim2]);
  } else {
    std::cerr << "Unsupported reshape instruction: " << param.opcode()
              << std::endl;
    exit(1);
  }

  std::cerr << "Permute order: " << order[0] << " " << order[1] << " "
            << order[2] << " " << order[3] << std::endl;

  std::vector<int> permuted_shape(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    permuted_shape[i] = input_shape[order[i]];
  }
  std::vector<int> permuted_strides = compute_strides(permuted_shape);

  std::vector<int> indices(input_shape.size(), 0);
  for (int i = 0; i < size; ++i) {
    std::vector<int> permuted_indices(order.size());
    for (size_t j = 0; j < order.size(); ++j) {
      permuted_indices[j] = indices[order[j]];
    }

    int permuted_index =
        compute_linear_index(permuted_indices, permuted_strides);

    outputs[permuted_index] = inputs[i];

    // Update multi-dimensional indices
    for (int j = input_shape.size() - 1; j >= 0; --j) {
      indices[j]++;
      if (indices[j] < input_shape[j]) {
        break;
      }
      indices[j] = 0;
    }
  }

  return outputs;
}
