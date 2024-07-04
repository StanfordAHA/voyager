#pragma once

#include "test/common/operations/Common.h"

template <typename T>
inline T *max_pooling(const T *inputs, const std::vector<int> shape) {
  T *outputs = new ACC_T[X * Y * K];
  for (int y = 0; y < Y / 2; y++) {
    for (int x = 0; x < X / 2; x++) {
      for (int k = 0; k < K; k++) {
        std::vector<T> v;

        for (int x_window = 0; x_window < 2; x_window++) {
          for (int y_window = 0; y_window < 2; y_window++) {
            int x_maxpool = x * 2 + x_window;
            int y_maxpool = y * 2 + y_window;
            v.push_back(outputs[y_maxpool * X * K + x_maxpool * K + k]);
            std::cout << outputs[y_maxpool * X * K + x_maxpool * K + k] << " ";
          }
        }

        T max = *std::max_element(v.begin(), v.end());
        std::cout << std::endl << "Max: " << max << std::endl;
      }
    }
  }
}
