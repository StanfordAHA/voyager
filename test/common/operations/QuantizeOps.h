#pragma once
#define NO_SYSC

#include <vector>

// clang-format off
#include "src/DataTypes.h"
// clang-format on

#include "src/ArchitectureParams.h"
#include "test/common/VerificationTypes.h"
#include "test/compiler/proto/param.pb.h"

template <typename TYPE, typename QUANTIZED_TYPE>
QUANTIZED_TYPE* quantize(std::any input, std::any scale, int size) {
  TYPE* input_tensor = std::any_cast<TYPE*>(input);

  TYPE scale_val = std::any_cast<TYPE>(scale);

  QUANTIZED_TYPE* quantized_output = new QUANTIZED_TYPE[size];

  for (int i = 0; i < size; i++) {
    TYPE scaled_val = input_tensor[i] / scale_val;

    quantized_output[i].int_val =
        scaled_val.float_val
            .template convert_to_ac_int<QUANTIZED_TYPE::ac_int_rep::width,
                                        QUANTIZED_TYPE::ac_int_rep::sign>();
    if (i == 0) {
      std::cout << "input_tensor[i]: " << input_tensor[i] << std::endl;
      std::cout << "scale_val: " << scale_val << std::endl;
      std::cout << "scaled_val: " << scaled_val << std::endl;
      std::cout << "quantized_output[i].int_val: "
                << quantized_output[i].int_val << std::endl;
    }
  }

  return quantized_output;
}