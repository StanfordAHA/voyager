#pragma once
#include <iostream>
#include <map>
#include <string>

#include "test/compiler/proto/param.pb.h"

// By default we have 2MB of SRAM per MINOTAUR SoC
// organized as 8x 256KB Banks with 2x 128KB Macros each
// FIXME: use a larger memory partition for testing
// #ifndef NUM_SRAM_BANKS
// #define NUM_SRAM_BANKS 8
// #endif
// #define SRAM_MEMORY_SIZE (NUM_SRAM_BANKS * 256 * 1024)
#ifndef NUM_SRAM_BANKS
#define NUM_SRAM_BANKS 256
#endif
#define SRAM_MEMORY_SIZE (NUM_SRAM_BANKS * 1024 * 1024)

// By default we have 12MB of RRAM per MINOTAUR SoC
// organized as 12x 1MB Banks with 4x 256KB Macros each
// 3x Superbanks and 4x Subbanks per Superbank
#ifndef NUM_RRAM_BANKS
#define NUM_RRAM_BANKS 64
#endif
#define RRAM_MEMORY_SIZE (NUM_RRAM_BANKS * 1024 * 1024)

// Size of the reference memory used for verification in MB
// Should be at least as large as the SRAM memory
#define REFERENCE_MEMORY_SIZE (1024 * 1024 * 8)

enum MemorySource { SRAM, RRAM };

inline std::ostream& operator<<(std::ostream& os, MemorySource& memory) {
  os << (memory == SRAM ? "SRAM" : "RRAM");
  return os;
}

// The accelerator has several interfaces for accessing memory, but
// these interfaces aren't always standard. For example, for some layers,
// an interface may access the residual, but for other layers, access weights.
// This data structure maps the accelerator interface to the MemorySource.
typedef std::map<std::string, MemorySource> AcceleratorMemoryMap;

struct Tiling {
  int loops[2][6];
  int x_loop_index[2];
  int y_loop_index[2];
  int reduction_loop_index[2];
  int weight_loop_index[2];
  int fx_index;
  int fy_index;
  int weight_reuse_index[2];
  int stride;
  bool replication;
};

inline std::ostream& operator<<(std::ostream& os, const Tiling& tiling) {
  os << "Loops: " << std::endl;
  for (int i = 0; i < 2; i++) {
    os << "  " << i << ": ";
    for (int j = 0; j < 6; j++) {
      os << tiling.loops[i][j] << " ";
    }
    os << std::endl;
  }
  os << "X Loop Index: " << tiling.x_loop_index[0] << " "
     << tiling.x_loop_index[1] << std::endl;
  os << "Y Loop Index: " << tiling.y_loop_index[0] << " "
     << tiling.y_loop_index[1] << std::endl;
  os << "Reduction Loop Index: " << tiling.reduction_loop_index[0] << " "
     << tiling.reduction_loop_index[1] << std::endl;
  os << "Weight Loop Index: " << tiling.weight_loop_index[0] << " "
     << tiling.weight_loop_index[1] << std::endl;
  os << "FX Index: " << tiling.fx_index << std::endl;
  os << "FY Index: " << tiling.fy_index << std::endl;
  os << "Weight Reuse Index: " << tiling.weight_reuse_index[0] << " "
     << tiling.weight_reuse_index[1] << std::endl;
  os << "Stride: " << tiling.stride << std::endl;
  os << "Replication: " << tiling.replication << std::endl;
  return os;
}

inline Tiling get_conv_tiling(codegen::MatrixParam param) {
  // input shape = (B, C, H, W)
  // weight shape = (OC, IC, KH, KW)
  int IC = param.input().shape(1);
  int IH = param.input().shape(2);
  int IW = param.input().shape(3);
  int OC = param.weight().shape(0);
  int KH = param.weight().shape(2);
  int KW = param.weight().shape(3);
  int STRIDE = param.stride(0);

  // TODO: loop tilings are hardcoded for now, ideally we would integrate with
  // a DSE tool like Interstellar

  if (IH == 224 && IW == 224 && IC == 3 && KH == 7 && KW == 7 &&
      OC == 64) {  // conv1
    const int fx = IC_DIMENSION == 16 ? 2 : (IC_DIMENSION == 32 ? 1 : 7);
    return {
        .loops = {{7, 7, 2, 1, 1, 1}, {1, 2, 7, fx, 16, 16}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
        .replication = true,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 64) {  // layer1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 64) {  // layer1_0_conv1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {4, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer1_x_conv3
    return {
        .loops = {{2, 2, 16, 1, 1, 1}, {4, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 64) {  // layer1_x_conv1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {16, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 128) {  // layer2_0_downsample (resnet18)
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {4, 2, 1, 1, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 512) {  // layer2_0_downsample (resnet50)
    return {
        .loops = {{2, 2, 16, 1, 1, 1}, {16, 2, 1, 1, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 128 && STRIDE == 2) {  // layer2_0_conv1
    return {
        .loops = {{4, 4, 4, 1, 1, 1}, {4, 3, 3, 2, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer2
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 128 && STRIDE == 1) {  // layer2_0_conv1
    return {
        .loops = {{2, 2, 8, 1, 1, 1}, {16, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 128) {  // layer2_x_conv1
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {32, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 56 && IW == 56 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer2_x_conv2
    return {
        .loops = {{2, 2, 8, 1, 1, 1}, {8, 1, 3, 3, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 1 && KW == 1 &&
             OC == 512) {  // layer2_x_conv3
    return {
        .loops = {{1, 1, 32, 1, 1, 1}, {8, 1, 1, 1, 28, 28}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_0_downsample (resnet18)
    return {
        .loops = {{2, 2, 2, 1, 1, 1}, {8, 1, 1, 8, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 1024 && STRIDE == 2) {  // layer3_0_downsample (resnet50)
    return {
        .loops = {{2, 2, 8, 1, 1, 1}, {32, 1, 1, 8, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 256 && STRIDE == 2) {  // layer3_0_conv1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {8, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 256) {  // layer3_0_conv2
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {16, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_0_conv1
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {32, 1, 1, 4, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_x_conv1
    return {
        .loops = {{1, 1, 4, 1, 1, 1}, {64, 1, 1, 4, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 256) {  // layer3_x_conv2
    return {
        .loops = {{1, 1, 4, 1, 1, 1}, {16, 3, 3, 4, 14, 14}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 1024) {  // layer3_x_conv3
    return {
        .loops = {{2, 2, 16, 1, 1, 1}, {16, 1, 1, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 28 && IW == 28 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer3
    return {
        .loops = {{2, 2, 4, 1, 1, 1}, {8, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 2) {  // layer4_0_downsample (resnet18)
    return {
        .loops = {{1, 1, 2, 1, 1, 1}, {16, 1, 1, 16, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 2048 && STRIDE == 2) {  // layer4_0_downsample (resnet50)
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {64, 1, 1, 16, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 512 && STRIDE == 2) {  // layer4_0_conv1
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {16, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 1) {  // layer4_0_conv1
    return {
        .loops = {{2, 2, 8, 1, 1, 1}, {64, 1, 1, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 7 && IW == 7 && IC == 2048 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 1) {  // layer4_x_conv1
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {128, 1, 1, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 14 && IW == 14 && IC == 512 && KH == 3 && KW == 3 &&
             OC == 512 && STRIDE == 2) {  // layer4_x_conv2
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {32, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 7 && IW == 7 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 2048) {  // layer4_x_conv3
    return {
        .loops = {{1, 1, 32, 1, 1, 1}, {32, 1, 1, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else if (IH == 7 && IW == 7 && IC == 512 && KH == 3 && KW == 3 &&
             OC == 512) {  // layer4
    return {
        .loops = {{1, 1, 8, 1, 1, 1}, {32, 3, 3, 4, 7, 7}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 3},
        .fx_index = 2,
        .fy_index = 1,
        .weight_reuse_index = {4, 5},
        .stride = STRIDE,
    };
  } else {
    std::cout << "Tiling has not been hardcoded for this layer yet"
              << std::endl;
    std::cout << "IH: " << IH << " IW: " << IW << " IC: " << IC << " KH: " << KH
              << " KW: " << KW << " OC: " << OC << " STRIDE: " << STRIDE
              << std::endl;
    exit(1);
  }
}

inline Tiling get_linear_tiling(codegen::MatrixParam param) {
  auto repeated_field = param.input().shape();
  std::vector<int> input_shape(repeated_field.begin(), repeated_field.end());
  repeated_field = param.weight().shape();
  std::vector<int> weight_shape(repeated_field.begin(), repeated_field.end());

  if (input_shape[1] == 128 && input_shape[2] == 512 &&
      weight_shape[0] == 128) {
    return {
        .loops = {{2, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 64}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else if (input_shape[1] == 128 && input_shape[2] == 128 &&
             weight_shape[0] == 128) {
    return {
        .loops = {{2, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 64}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else if (input_shape[1] == 128 && input_shape[2] == 128 &&
             weight_shape[0] == 512) {
    return {
        .loops = {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else if (input_shape[1] == 128 && input_shape[2] == 768 &&
             weight_shape[0] == 768) {
    return {
        .loops = {{8, 1, 1, 1, 1, 1}, {48, 48, 1, 1, 1, 16}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else {
    std::cerr << "Unsupported inputs dimensions" << std::endl;
    std::cerr << "Input shape: ";
    for (int dim : input_shape) std::cerr << dim << " ";
    std::cerr << std::endl;
    std::cerr << "Weight shape: ";
    for (int dim : weight_shape) std::cerr << dim << " ";
    std::cerr << std::endl;
    exit(1);
  }
}

inline Tiling get_matmul_tiling(codegen::MatrixParam param) {
  auto repeated_field = param.input().shape();
  std::vector<int> input_shape(repeated_field.begin(), repeated_field.end());
  repeated_field = param.weight().shape();
  std::vector<int> weight_shape(repeated_field.begin(), repeated_field.end());
  const int ndim = input_shape.size();

  if (input_shape[ndim - 2] == 128 && input_shape[ndim - 1] == 32 &&
      weight_shape[1] == 128) {
    return {
        .loops = {{4, 1, 1, 1, 1, 1}, {2, 8, 1, 1, 1, 32}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else if (input_shape[ndim - 2] == 128 && input_shape[ndim - 1] == 64 &&
             weight_shape[1] == 128) {
    return {
        .loops = {{4, 1, 1, 1, 1, 1}, {4, 8, 1, 1, 1, 32}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else if (input_shape[ndim - 2] == 128 && input_shape[ndim - 1] == 128 &&
             weight_shape[1] == 32) {
    return {
        .loops = {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},
        .x_loop_index = {0, 5},
        .y_loop_index = {1, 4},
        .reduction_loop_index = {3, 0},
        .weight_loop_index = {2, 1},
        .fx_index = 3,
        .fy_index = 2,
        .weight_reuse_index = {4, 5},
        .stride = 1,
        .replication = false,
    };
  } else {
    std::cerr << "Unsupported inputs dimensions" << std::endl;
    std::cerr << "Input shape: ";
    for (int dim : input_shape) std::cerr << dim << " ";
    std::cerr << std::endl;
    std::cerr << "Weight shape: ";
    for (int dim : weight_shape) std::cerr << dim << " ";
    std::cerr << std::endl;
    exit(1);
  }
}

inline Tiling get_conv2d_tiling(codegen::AcceleratorParam param) {
  const auto matrix_param = param.matrix_param();
  const auto input_shape = matrix_param.input().shape();
  const auto weight_shape = matrix_param.weight().shape();
  const auto output_shape = param.output().shape();

  int X = output_shape[2];
  int Y = output_shape[3];
  int C = weight_shape[1];
  int K = weight_shape[0];

  int fx = weight_shape[2];
  int fy = weight_shape[3];
  int k0 = std::min(K / OC_DIMENSION, 1024 / OC_DIMENSION / fx / fy);
  int k1 = K / k0;
  int x0 = std::sqrt(1024 / k0);  // assume x0 == y0
  int x1 = X / x0;
  int y0 = x0;
  int y1 = x1;
  int c0 = C / IC_DIMENSION;

  return {
      .loops = {{x1, y1, k1, 1, 1, 1}, {c0, k0, fx, fy, y0, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = matrix_param.stride(0),
      .replication = false,
  };
}

inline Tiling get_pooling_tiling(codegen::AcceleratorParam param) {
  const auto pooling_param = param.pooling_param();
  const auto input_shape = pooling_param.input().shape();
  const auto output_shape = param.output().shape();

  int X = input_shape[2];
  int Y = input_shape[3];
  int C = pooling_param.input().shape(1);
  int K = param.output().shape(1);

  int x0, y0, stride;
  if (pooling_param.output_size_size() > 0) {
    int output_h = pooling_param.output_size(0);
    int output_w = pooling_param.output_size(1);
    stride = X / output_h;
    x0 = X - (output_h - 1) * stride;
    y0 = Y - (output_w - 1) * stride;
  } else {
    x0 = pooling_param.kernel_size(0);
    y0 = pooling_param.kernel_size(1);
    stride = pooling_param.stride(0);
  }

  int x1 = X / x0;
  int y1 = Y / y0;
  int c0 = C / IC_DIMENSION;
  int k0 = K / OC_DIMENSION;

  return {
      .loops = {{x1, y1, 1, 1, 1, 1}, {c0, k0, 1, 1, y0, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = stride,
      .replication = false,
  };
}

inline Tiling get_gemm_tiling(codegen::AcceleratorParam param) {
  const auto matrix_param = param.matrix_param();
  const auto input_shape = matrix_param.input().shape();
  const auto weight_shape = matrix_param.weight().shape();

  int X = 1;
  for (int i = 1; i < input_shape.size() - 1; i++) {
    X *= input_shape[i];
  }
  int C = weight_shape[1];
  int K = weight_shape[0];

  int k0 = K / OC_DIMENSION;
  int x0 = 1024 / k0;  // largest accumulation buffer size
  int x1 = X / x0;
  int c0 = C / IC_DIMENSION;

  return {
      .loops = {{x1, 1, 1, 1, 1, 1}, {c0, k0, 1, 1, 1, x0}},
      .x_loop_index = {0, 5},
      .y_loop_index = {1, 4},
      .reduction_loop_index = {3, 0},
      .weight_loop_index = {2, 1},
      .fx_index = 3,
      .fy_index = 2,
      .weight_reuse_index = {4, 5},
      .stride = 1,
      .replication = false,
  };
}
