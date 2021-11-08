#include <locale>
#include <string>

#include "test/common/GoldModel.h"
#include "test/common/Harness.h"
#include "test/common/Utils.h"
#include "test/mobilebert/params.h"
#include "test/simple/params.h"

void run_test(Params params) {
  INPUT_DATATYPE *mainMemory = new INPUT_DATATYPE[4 * 1024 * 1024];

  // Create matrix A
  INPUT_DATATYPE *matrixA =
      new INPUT_DATATYPE[params.M0 * params.M1 * params.N1 * DIMENSION];
  for (int i = 0; i < params.M0 * params.M1; i++) {
    for (int j = 0; j < params.N1 * DIMENSION; j++) {
      int val = i * 10 + j;

      mainMemory[params.INPUT_OFFSET + i * (params.N1 * DIMENSION) + j] = val;
      matrixA[i * (params.N1 * DIMENSION) + j] = val;
    }
  }

  INPUT_DATATYPE *matrixB =
      new INPUT_DATATYPE[params.N1 * DIMENSION * params.P1 * params.P2 *
                         DIMENSION];
  for (int i = 0; i < params.N1 * DIMENSION; i++) {
    for (int j = 0; j < params.P1 * params.P2 * DIMENSION; j++) {
      int val = i;

      mainMemory[params.WEIGHT_OFFSET +
                 i * (params.P1 * params.P2 * DIMENSION) + j] = val;
      matrixB[i * (params.P1 * params.P2 * DIMENSION) + j] = val;
    }
  }

  OUTPUT_DATATYPE *matrixC =
      new OUTPUT_DATATYPE[params.M0 * params.M1 * params.P1 * params.P2 *
                          DIMENSION];

  run_op(params, mainMemory);
  run_gold_op(params, matrixA, matrixB, matrixC);
  compare_arrays(&mainMemory[params.OUTPUT_OFFSET], matrixC,
                 params.M0 * params.M1 * params.P1 * params.P2 * DIMENSION);

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
