#include <locale>
#include <string>

#include "test/common/GoldModel.h"
#include "test/common/Harness.h"
#include "test/common/Utils.h"
#include "test/mobilebert/params.h"
#include "test/simple/params.h"

void run_test(Params params) {
  INPUT_DATATYPE *mainMemory = new INPUT_DATATYPE[4 * 1024 * 1024];

  int X = params.loops[0][params.inputXLoopIndex[0]] *
          params.loops[1][params.inputXLoopIndex[1]];
  int Y = params.loops[0][params.inputYLoopIndex[0]] *
          params.loops[1][params.inputYLoopIndex[1]];
  int C = params.loops[1][params.reductionLoopIndex[1]] * DIMENSION;
  int K = params.loops[0][params.weightLoopIndex[0]] *
          params.loops[1][params.weightLoopIndex[1]] * DIMENSION;
  int FX = params.loops[1][params.fxIndex];
  int FY = params.loops[1][params.fyIndex];
  int STRIDE = params.STRIDE;

  // Create matrix A
  INPUT_DATATYPE *matrixA = new INPUT_DATATYPE[(STRIDE * X) * (STRIDE * Y) * C];
  for (int y = 0; y < STRIDE * Y; y++) {
    for (int x = 0; x < STRIDE * X; x++) {
      for (int c = 0; c < C; c++) {
        // int val = i * 10 + j;
        // int val = rand() % 128;
        int val = x;

        int address = y * (STRIDE * X) * C + x * C + c;

        mainMemory[params.INPUT_OFFSET + address] = val;
        matrixA[address] = val;
      }
    }
  }

  INPUT_DATATYPE *matrixB = new INPUT_DATATYPE[FX * FY * C * K];
  for (int fy = 0; fy < FY; fy++) {
    for (int fx = 0; fx < FX; fx++) {
      for (int c = 0; c < C; c++) {
        for (int k = 0; k < K; k++) {
          // int val = i;
          // int val = rand() % 128;
          int val = k;

          int address = fy * FX * C * K + fx * C * K + c * K + k;
          mainMemory[params.WEIGHT_OFFSET + address] = val;
          matrixB[address] = val;
        }
      }
    }
  }

  OUTPUT_DATATYPE *matrixC = new OUTPUT_DATATYPE[X * Y * K];

  run_op(params, mainMemory);
  run_gold_op(params, matrixA, matrixB, matrixC);
  compare_arrays(&mainMemory[params.OUTPUT_OFFSET], matrixC, X * Y * K);

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  delete[] mainMemory;
}

int sc_main(int argc, char *argv[]) {
  Params params = simple;

  const char *testName = std::getenv("TEST");
  if (testName) {
    std::string test(testName);
    std::cout << "Running test: " << test << std::endl;

    if (test == "simple") {
      params = simple;
    } else if (test == "conv") {
      params = conv;
    } else if (test == "inputBottleneck") {
      params = inputBottleneck;
    } else if (test == "qkvProjection") {
      params = qkvProjection;
    } else if (test == "qkAttention") {
      params = qkAttention;
    } else if (test == "vAttention") {
      params = vAttention;
    } else if (test == "wProjection") {
      params = wProjection;
    } else if (test == "ffn1") {
      params = ffn1;
    } else if (test == "ffn2") {
      params = ffn2;
    } else if (test == "outputBottleneck") {
      params = outputBottleneck;
    } else {
      std::cout << "Warning! Test not found!" << std::endl;
    }
  } else {
    std::cout << "Warning! No test specified! Please set the environment "
                 "variable TEST"
              << std::endl;
  }

  run_test(params);
  return 0;
}
