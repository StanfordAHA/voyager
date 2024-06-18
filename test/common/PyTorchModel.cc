#include "test/common/PyTorchModel.h"

#include <algorithm>
#include <cstring>
#include <vector>

inline Tiling GetTiling(example::MatrixParam param) {
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
    return {.loops = {{7, 7, 2, 1, 1, 1}, {1, 2, 7, 2, 16, 16}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE,
            .replication = true};
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 64) {  // layer1
    return {.loops = {{2, 2, 4, 1, 1, 1}, {4, 1, 3, 3, 28, 28}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 64) {  // layer1_0_conv1
    return {.loops = {{2, 2, 4, 1, 1, 1}, {4, 1, 1, 1, 28, 28}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer1_x_conv3
    return {.loops = {{2, 2, 16, 1, 1, 1}, {4, 1, 1, 1, 28, 28}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 64) {  // layer1_x_conv1
    return {.loops = {{2, 2, 4, 1, 1, 1}, {16, 1, 1, 1, 28, 28}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 1 && KW == 1 &&
             OC == 128) {  // layer2_0_downsample (resnet18)
    return {.loops = {{2, 2, 4, 1, 1, 1}, {4, 2, 1, 1, 14, 14}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 512) {  // layer2_0_downsample (resnet50)
    return {.loops = {{2, 2, 16, 1, 1, 1}, {16, 2, 1, 1, 14, 14}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 56 && IW == 56 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 128 && STRIDE == 2) {  // layer2_0_conv1
    return {.loops = {{4, 4, 4, 1, 1, 1}, {4, 3, 3, 2, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer2
    return {.loops = {{1, 1, 8, 1, 1, 1}, {8, 1, 3, 3, 28, 28}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 56 && IW == 56 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 128 && STRIDE == 1) {  // layer2_0_conv1
    return {.loops = {{2, 2, 8, 1, 1, 1}, {16, 1, 1, 1, 28, 28}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 128) {  // layer2_x_conv1
    return {.loops = {{1, 1, 8, 1, 1, 1}, {32, 1, 1, 1, 28, 28}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 56 && IW == 56 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer2_x_conv2
    return {.loops = {{2, 2, 8, 1, 1, 1}, {8, 1, 3, 3, 14, 14}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 1 && KW == 1 &&
             OC == 512) {  // layer2_x_conv3
    return {.loops = {{1, 1, 32, 1, 1, 1}, {8, 1, 1, 1, 28, 28}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 1},
            .fx_index = 3,
            .fy_index = 2,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_0_downsample (resnet18)
    return {.loops = {{2, 2, 2, 1, 1, 1}, {8, 1, 1, 8, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 1024 && STRIDE == 2) {  // layer3_0_downsample (resnet50)
    return {.loops = {{2, 2, 8, 1, 1, 1}, {32, 1, 1, 8, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 128 && KH == 3 && KW == 3 &&
             OC == 256 && STRIDE == 2) {  // layer3_0_conv1
    return {.loops = {{2, 2, 4, 1, 1, 1}, {8, 3, 3, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 256) {  // layer3_0_conv2
    return {.loops = {{2, 2, 4, 1, 1, 1}, {16, 3, 3, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_0_conv1
    return {.loops = {{2, 2, 4, 1, 1, 1}, {32, 1, 1, 4, 14, 14}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 256) {  // layer3_x_conv1
    return {.loops = {{1, 1, 4, 1, 1, 1}, {64, 1, 1, 4, 14, 14}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 256) {  // layer3_x_conv2
    return {.loops = {{1, 1, 4, 1, 1, 1}, {16, 3, 3, 4, 14, 14}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 1024) {  // layer3_x_conv3
    return {.loops = {{2, 2, 16, 1, 1, 1}, {16, 1, 1, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 28 && IW == 28 && IC == 64 && KH == 3 && KW == 3 &&
             OC == 128) {  // layer3
    return {.loops = {{2, 2, 4, 1, 1, 1}, {8, 3, 3, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 2) {  // layer4_0_downsample (resnet18)
    return {.loops = {{1, 1, 2, 1, 1, 1}, {16, 1, 1, 16, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 2048 && STRIDE == 2) {  // layer4_0_downsample (resnet50)
    return {.loops = {{1, 1, 8, 1, 1, 1}, {64, 1, 1, 16, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 14 && IW == 14 && IC == 256 && KH == 3 && KW == 3 &&
             OC == 512 && STRIDE == 2) {  // layer4_0_conv1
    return {.loops = {{1, 1, 8, 1, 1, 1}, {16, 3, 3, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 14 && IW == 14 && IC == 1024 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 1) {  // layer4_0_conv1
    return {.loops = {{2, 2, 8, 1, 1, 1}, {64, 1, 1, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 7 && IW == 7 && IC == 2048 && KH == 1 && KW == 1 &&
             OC == 512 && STRIDE == 1) {  // layer4_x_conv1
    return {.loops = {{1, 1, 8, 1, 1, 1}, {128, 1, 1, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 14 && IW == 14 && IC == 512 && KH == 3 && KW == 3 &&
             OC == 512 && STRIDE == 2) {  // layer4_x_conv2
    return {.loops = {{1, 1, 8, 1, 1, 1}, {32, 3, 3, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 7 && IW == 7 && IC == 512 && KH == 1 && KW == 1 &&
             OC == 2048) {  // layer4_x_conv3
    return {.loops = {{1, 1, 32, 1, 1, 1}, {32, 1, 1, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else if (IH == 7 && IW == 7 && IC == 512 && KH == 3 && KW == 3 &&
             OC == 512) {  // layer4
    return {.loops = {{1, 1, 8, 1, 1, 1}, {32, 3, 3, 4, 7, 7}},
            .x_loop_index = {0, 5},
            .y_loop_index = {1, 4},
            .reduction_loop_index = {3, 0},
            .weight_loop_index = {2, 3},
            .fx_index = 2,
            .fy_index = 1,
            .weight_reuse_index = {4, 5},
            .stride = STRIDE};
  } else {
    std::cout << "Tiling has not been hardcoded for this layer yet"
              << std::endl;
    std::cout << "IH: " << IH << " IW: " << IW << " IC: " << IC << " KH: " << KH
              << " KW: " << KW << " OC: " << OC << " STRIDE: " << STRIDE
              << std::endl;
    exit(1);
  }
}

/***************************************************************************
 * FusedMultiplyAdd Functions
 *
 * These inline functions handle the fused multiply-add operation.
 ***************************************************************************/

inline void FusedMultiplyAdd(float a, float b, float &c) { c += a * b; }

inline void FusedMultiplyAdd(INPUT_DATATYPE a, INPUT_DATATYPE b,
                             ACCUM_DATATYPE::AccumulationDatatype &c) {
#ifdef HYBRID_FP8
  HYBRID_TYPE hybrid_a(a);
  HYBRID_TYPE hybrid_b(b);
  c = hybrid_a.fma(hybrid_b, c);
#else
  INPUT_DATATYPE::AccumulationDatatype v1 = a;
  INPUT_DATATYPE::AccumulationDatatype v2 = b;
  c = v1.fma(v2, c);
#endif
}

#ifndef NO_UNIVERSAL
inline void FusedMultiplyAdd(UniversalPosit a, UniversalPosit b,
                             UniversalPositAccum &c) {
  UniversalPositAccum product;
  sw::universal::value<15> internal = sw::universal::fma<8, 1>(a, b, 0);
  sw::universal::convert<16, 1, 15>(internal, product);
  c += product;
}
#endif

inline void ReLU(float &a) { a = a > 0 ? a : 0; }

inline void ReLU(ACCUM_DATATYPE::AccumulationDatatype &a) { a.relu(); }

#ifndef NO_UNIVERSAL
inline void ReLU(UniversalPositAccum &a) { a = a > 0 ? a : 0; }
#endif

/***************************************************************************
 * ReadTensor Functions
 *
 * These inline functions handle reading tensor values from different types
 * of matrices. The functions manage whether the data should be read in
 * single or double precision formats. There are different overloads to
 * support various data types including float, INPUT_DATATYPE, and
 * UniversalPosit (if enabled).
 ***************************************************************************/

inline float ReadTensor(float *matrix, int index,
                        bool double_precision = false) {
  return double_precision ? matrix[2 * index] : matrix[index];
}

inline ACCUM_DATATYPE ReadTensor(INPUT_DATATYPE *matrix, int index,
                                 bool double_precision = false) {
  if (!double_precision) {
    return static_cast<ACCUM_DATATYPE>(matrix[index]);
  } else {
    return ACCUM_DATATYPE(&matrix[2 * index]);
  }
}

#ifndef NO_UNIVERSAL
inline UniversalPositAccum ReadTensor(UniversalPosit *matrix, int index,
                                      bool double_precision = false) {
  if (!double_precision) {
    return static_cast<UniversalPositAccum>(matrix[index]);
  }

  int encoding1 = matrix[2 * index].encoding();
  int encoding2 = matrix[2 * index + 1].encoding();
  UniversalPositAccum p16;
  p16.setbits((encoding2 << 8) + encoding1);
  return p16;
}
#endif

/***************************************************************************
 * SaveTensor Functions
 *
 * These inline functions handle saving tensor values into different types
 * of matrices. The functions manage whether the data should be stored in
 * single or double precision formats. There are different overloads to
 * support various data types including float, INPUT_DATATYPE, and
 * UniversalPosit (if enabled).
 ***************************************************************************/

inline void SaveTensor(float *matrix, int index, float value,
                       bool double_precision = false) {
  if (!double_precision) {
    matrix[index] = value;
  } else {
    matrix[2 * index] = value;
    matrix[2 * index + 1] = 0;
  }
}

inline void SaveTensor(INPUT_DATATYPE *matrix, int index,
                       ACCUM_DATATYPE::AccumulationDatatype value,
                       bool double_precision = false) {
  if (!double_precision) {
    matrix[index] = static_cast<INPUT_DATATYPE>(value);
  } else {
    ACCUM_DATATYPE p16 = value;
    p16.storeAsLowerPrecision(&matrix[2 * index]);
  }
}

#ifndef NO_UNIVERSAL
inline void SaveTensor(UniversalPosit *matrix, int index,
                       UniversalPositAccum value,
                       bool double_precision = false) {
  if (!double_precision) {
    matrix[index] = static_cast<UniversalPosit>(value);
  } else {
    int bits = value.encoding();
    matrix[2 * index].setbits(bits & 0xFF);
    matrix[2 * index + 1].setbits((bits >> 8) & 0xFF);
  }
}
#endif

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T>
int GEMM(Tiling tiling, const std::vector<INPUT_T *> args,
         ACCUMULATE_T *&output_matrix) {
  std::cerr << "GEMM" << std::endl;
  std::cerr << tiling << std::endl;

  const int X = tiling.loops[0][tiling.x_loop_index[0]] *
                tiling.loops[1][tiling.x_loop_index[1]];
  const int Y = tiling.loops[0][tiling.y_loop_index[0]] *
                tiling.loops[1][tiling.y_loop_index[1]];
  const int C = tiling.replication
                    ? 3
                    : tiling.loops[1][tiling.reduction_loop_index[1]] * 16;
  const int K = tiling.loops[0][tiling.weight_loop_index[0]] *
                tiling.loops[1][tiling.weight_loop_index[1]] * 16;
  const int FX = tiling.replication ? 7 : tiling.loops[1][tiling.fx_index];
  const int FY = tiling.loops[1][tiling.fy_index];
  const int stride = tiling.stride;

  // adjust loop counters for dimension != 16
  if (IC_DIMENSION < 16) {
    tiling.loops[1][tiling.reduction_loop_index[1]] *= (16 / IC_DIMENSION);
  } else if (IC_DIMENSION > 16) {
    tiling.loops[1][tiling.reduction_loop_index[1]] /= (IC_DIMENSION / 16);
  }

  if (OC_DIMENSION < 16) {
    tiling.loops[0][tiling.weight_loop_index[0]] *= (16 / OC_DIMENSION);
  } else if (OC_DIMENSION > 16) {
    // if the inner weight loop is >=4, we should reduce the inner loop
    // (otherwise, we violate the weight buffer constraint) otherwise, we
    // reduce the outer loop
    if ((tiling.loops[1][tiling.weight_loop_index[1]] >= 4 &&
         tiling.loops[1][tiling.fx_index] > 1 &&
         tiling.loops[1][tiling.fy_index] > 1)) {
      tiling.loops[1][tiling.weight_loop_index[1]] /= (OC_DIMENSION / 16);
    } else if (tiling.loops[0][tiling.weight_loop_index[0]] <
                   (OC_DIMENSION / 16) &&
               tiling.loops[0][tiling.weight_loop_index[0]] != 1) {
      const int reduction_factor =
          OC_DIMENSION / 16 / tiling.loops[0][tiling.weight_loop_index[0]];
      tiling.loops[0][tiling.weight_loop_index[0]] = 1;
      tiling.loops[1][tiling.weight_loop_index[1]] /= reduction_factor;
    } else if (tiling.loops[0][tiling.weight_loop_index[0]] == 1) {
      tiling.loops[1][tiling.weight_loop_index[1]] /= (OC_DIMENSION / 16);
    } else {
      tiling.loops[0][tiling.weight_loop_index[0]] /= (OC_DIMENSION / 16);
    }
  }

  int X0 = tiling.loops[1][tiling.x_loop_index[1]];
  int Y0 = tiling.loops[1][tiling.y_loop_index[1]];
  int K0 = tiling.loops[1][tiling.weight_loop_index[1]];
  int IC_unroll = IC_DIMENSION;

  if (tiling.replication) {
    tiling.loops[1][tiling.fx_index] = 7;
    IC_unroll = 3;
    tiling.loops[1][tiling.reduction_loop_index[1]] = 1;
  }

  // assert that none of tiling.loops are 0
  for (int j = 0; j < 3; j++) {
    assert(tiling.loops[0][j] != 0);
  }
  for (int j = 0; j < 6; j++) {
    assert(tiling.loops[1][j] != 0);
  }

  // HACK: initialize output matrix and create array at the same place
  output_matrix = new ACCUMULATE_T[X * Y * K];

  if (args[2] != nullptr) {
    for (int y = 0; y < Y; y++) {
      for (int x = 0; x < X; x++) {
        for (int k = 0; k < K; k++) {
          int addr = y * X * K + x * K + k;
          output_matrix[addr] = ReadTensor(args[2], k, true);
        }
      }
    }
  }

  int counters[2][6] = {0};
  for (counters[0][0] = 0; counters[0][0] < tiling.loops[0][0];
       counters[0][0]++) {
    for (counters[0][1] = 0; counters[0][1] < tiling.loops[0][1];
         counters[0][1]++) {
      for (counters[0][2] = 0; counters[0][2] < tiling.loops[0][2];
           counters[0][2]++) {
        int x1 = counters[0][tiling.x_loop_index[0]];
        int y1 = counters[0][tiling.y_loop_index[0]];
        int k1 = counters[0][tiling.weight_loop_index[0]];

        for (counters[1][0] = 0; counters[1][0] < tiling.loops[1][0];
             counters[1][0]++) {
          for (counters[1][1] = 0; counters[1][1] < tiling.loops[1][1];
               counters[1][1]++) {
            for (counters[1][2] = 0; counters[1][2] < tiling.loops[1][2];
                 counters[1][2]++) {
              for (counters[1][3] = 0; counters[1][3] < tiling.loops[1][3];
                   counters[1][3]++) {
                for (counters[1][4] = 0; counters[1][4] < tiling.loops[1][4];
                     counters[1][4]++) {
                  for (counters[1][5] = 0; counters[1][5] < tiling.loops[1][5];
                       counters[1][5]++) {
                    int x0 = counters[1][tiling.x_loop_index[1]];
                    int y0 = counters[1][tiling.y_loop_index[1]];
                    int c0 = counters[1][tiling.reduction_loop_index[1]];
                    int k0 = counters[1][tiling.weight_loop_index[1]];
                    int fx = counters[1][tiling.fx_index] - (FX - 1) / 2;
                    int fy = counters[1][tiling.fy_index] - (FY - 1) / 2;

                    int x = x1 * X0 + x0;
                    int y = y1 * Y0 + y0;

                    for (int oc0 = 0; oc0 < OC_DIMENSION; oc0++) {
                      int k = (k1 * K0 + k0) * OC_DIMENSION + oc0;
                      int output_addr = y * X * K + x * K + k;

                      for (int ic0 = 0; ic0 < IC_unroll; ic0++) {
                        int c = c0 * IC_unroll + ic0;
                        int input_addr = (stride * y + fy) * stride * X * C +
                                         (stride * x + fx) * C + c;
                        int weight_addr = (fy + (FY - 1) / 2) * FX * C * K +
                                          (fx + (FX - 1) / 2) * C * K + c * K +
                                          k;
                        if (stride * x + fx >= 0 &&
                            stride * x + fx < stride * X &&
                            stride * y + fy >= 0 &&
                            stride * y + fy < stride * Y) {
                          INTERMEDIATE_T input =
                              ReadTensor(args[0], input_addr);
                          INTERMEDIATE_T weight =
                              ReadTensor(args[1], weight_addr);
                          FusedMultiplyAdd(input, weight,
                                           output_matrix[output_addr]);
                        }
                      }
                      if (tiling.replication) {
                        if (IC_DIMENSION == 16) {
                          if (counters[1][tiling.fx_index] == 3 ||
                              counters[1][tiling.fx_index] == 6) {
                            output_matrix[output_addr] =
                                static_cast<INTERMEDIATE_T>(
                                    output_matrix[output_addr]);
                          }
                        } else if (IC_DIMENSION == 32) {
                          if (counters[1][tiling.fx_index] == 6) {
                            output_matrix[output_addr] =
                                static_cast<INTERMEDIATE_T>(
                                    output_matrix[output_addr]);
                          }
                        }
                      } else {
                        output_matrix[output_addr] =
                            static_cast<INTERMEDIATE_T>(
                                output_matrix[output_addr]);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return X * Y * K;
}

// Function to check if two shapes are broadcastable
bool are_broadcastable(const std::vector<int> &shape1,
                       const std::vector<int> &shape2) {
  int n1 = shape1.size();
  int n2 = shape2.size();
  int min_size = std::min(n1, n2);

  for (int i = 1; i <= min_size; i++) {
    int dim1 = n1 - i >= 0 ? shape1[n1 - i] : 1;
    int dim2 = n2 - i >= 0 ? shape2[n2 - i] : 1;

    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      return false;
    }
  }
  return true;
}

// Function to compute the broadcasted shape
std::vector<int> broadcast_shape(const std::vector<int> &shape1,
                                 const std::vector<int> &shape2) {
  if (!are_broadcastable(shape1, shape2)) {
    throw std::invalid_argument("Shapes are not broadcastable");
  }

  int n1 = shape1.size();
  int n2 = shape2.size();
  int max_size = std::max(n1, n2);
  std::vector<int> result_shape(max_size);

  for (int i = 1; i <= max_size; i++) {
    int dim1 = n1 - i >= 0 ? shape1[n1 - i] : 1;
    int dim2 = n2 - i >= 0 ? shape2[n2 - i] : 1;
    result_shape[max_size - i] = std::max(dim1, dim2);
  }

  return result_shape;
}

// Recursive function to add tensors with broadcasting
template <typename T>
void perform_elwise_op_recursivly(const T *tensor1,
                                  const std::vector<int> &shape1,
                                  const T *tensor2,
                                  const std::vector<int> &shape2, T *result,
                                  const std::vector<int> &result_shape,
                                  std::string operation, int dim, int offset1,
                                  int offset2, int offset_res) {
  if (dim == result_shape.size()) {
    if (operation == "add" || operation == "add_") {
      result[offset_res] = tensor1[offset1] + tensor2[offset2];
    } else if (operation == "sub" || operation == "sub_") {
      result[offset_res] = tensor1[offset1] - tensor2[offset2];
    } else if (operation == "mul" || operation == "mul_") {
      result[offset_res] = tensor1[offset1] * tensor2[offset2];
    } else if (operation == "div" || operation == "div_") {
      result[offset_res] = tensor1[offset1] / tensor2[offset2];
    } else {
      throw std::invalid_argument("Invalid operation: " + operation);
    }
    return;
  }

  int stride1 = dim < shape1.size() ? shape1[dim] : 1;
  int size1 = 1;
  for (int i = dim + 1; i < shape1.size(); i++) {
    size1 *= shape1[i];
  }

  int stride2 = dim < shape2.size() ? shape2[dim] : 1;
  int size2 = 1;
  for (int i = dim + 1; i < shape2.size(); i++) {
    size2 *= shape2[i];
  }

  int stride_res = result_shape[dim];
  int size_res = 1;
  for (int i = dim + 1; i < result_shape.size(); i++) {
    size_res *= result_shape[i];
  }

  for (int i = 0; i < stride_res; i++) {
    perform_elwise_op_recursivly(
        tensor1, shape1, tensor2, shape2, result, result_shape, operation,
        dim + 1, offset1 + (stride1 == 1 ? 0 : i * size1),
        offset2 + (stride2 == 1 ? 0 : i * size2), offset_res + i * size_res);
  }
}

// Function to add two tensors with broadcasting
template <typename T>
T *perform_elwise_operation(const T *tensor1, const std::vector<int> &shape1,
                            const T *tensor2, const std::vector<int> &shape2,
                            std::string operation) {
  std::vector<int> result_shape = broadcast_shape(shape1, shape2);
  int result_size = 1;
  for (int dim : result_shape) {
    result_size *= dim;
  }

  T *result = new T[result_size];
  perform_elwise_op_recursivly(tensor1, shape1, tensor2, shape2, result,
                               result_shape, operation, 0, 0, 0, 0);

  return result;
}

const std::set<std::string> activations = {"relu", "relu_", "gelu", "gelu_"};
const std::set<std::string> arithmetics = {"add", "add_", "sub", "sub_",
                                           "mul", "mul_", "div", "div_"};

template <typename INPUT_T, typename ACCUMULATE_T, typename INTERMEDIATE_T>
void run_pytorch_op(const example::AcceleratorParam param,
                    std::vector<INPUT_T *> args) {
  int arg_index = 0;
  int output_size;
  ACCUMULATE_T *output_matrix = nullptr;

  if (param.has_matrix_param()) {
    const auto &matrix_param = param.matrix_param();
    Tiling tiling = GetTiling(matrix_param);
    output_size = GEMM<INPUT_T, ACCUMULATE_T, INTERMEDIATE_T>(tiling, args,
                                                              output_matrix);
    arg_index = 3;
  }

  for (const auto &vector_param : param.vector_params()) {
    std::cerr << "vector_param: " << vector_param.opcode() << std::endl;
    if (activations.find(vector_param.opcode()) != activations.end()) {
      auto repeated_field = vector_param.input().shape();
      std::vector<int> shape(repeated_field.begin(), repeated_field.end());

      int size = 1;
      for (int dim : shape) {
        size *= dim;
      }

      for (int i = 0; i < size; i++) {
        ReLU(output_matrix[i]);
      }
    } else if (arithmetics.find(vector_param.opcode()) != arithmetics.end()) {
      auto repeated_field_1 = vector_param.input().shape();
      std::vector<int> shape1(repeated_field_1.begin(), repeated_field_1.end());
      auto repeated_field_2 = vector_param.input().shape();
      std::vector<int> shape2(repeated_field_2.begin(), repeated_field_2.end());

      int other_size = 1;
      for (int dim : shape1) {
        other_size *= dim;
      }

      ACCUMULATE_T *other_matrix = new ACCUMULATE_T[other_size];
      for (int j = 0; j < other_size; j++) {
        other_matrix[j] = ReadTensor(args[arg_index], j);
      }

      output_matrix = perform_elwise_operation(
          output_matrix, shape1, other_matrix, shape2, vector_param.opcode());
    } else {
      std::cerr << "Unsupported opcode: " << vector_param.opcode() << std::endl;
      exit(1);
    }
  }

  for (int i = 0; i < output_size; i++) {
    SaveTensor(args.back(), i, output_matrix[i]);
  }
}

void run_pytorch_model(const example::AcceleratorParam &param,
                       std::vector<float *> args) {
  run_pytorch_op<float, float, float>(param, args);
}

void run_pytorch_model(const example::AcceleratorParam &param,
                       std::vector<INPUT_DATATYPE *> args) {
  run_pytorch_op<INPUT_DATATYPE, ACCUM_DATATYPE::AccumulationDatatype,
                 ACCUM_DATATYPE>(param, args);
}

#ifndef NO_UNIVERSAL
void run_pytorch_model(const example::AcceleratorParam &param,
                       std::vector<UniversalPosit *> args) {
  run_pytorch_op<UniversalPosit, UniversalPositAccum, UniversalPositAccum>(
      param, args);
}
#endif
