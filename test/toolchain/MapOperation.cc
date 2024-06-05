#include "test/toolchain/MapOperation.h"

#include "test/toolchain/operations/Operations.h"

// create Matrix and Vector Params from SimplifiedParams
void MapOperation(const SimplifiedParams &params, const MemoryMap &memoryMap,
                  std::deque<BaseParams *> &mappedParams,
                  std::deque<AcceleratorMemoryMap> &opMemoryMaps) {
  if (params.GRAD_CLIPPING_UNIT_TEST) {
    MapGradNormUnitTest(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.SOFTMAX) {
    MapSoftmax(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.SOFTMAX_GRAD) {
    MapSoftmaxGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.FC_GRAD) {
    MapFCGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.FC) {
    MapFC(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.NO_NORM) {
    MapNoNorm(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.NO_NORM_GRAD) {
    MapNoNormGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.MSE_GRAD || params.BCE_WITH_LOGITS_GRAD) {
    MapGenericErrorGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.BIAS_GRAD) {
    MapBiasGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.CROSS_ENTROPY_GRAD) {
    MapCrossEntropyGrad(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.WEIGHT_UPDATE) {
    MapWeightUpdate(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.NOP) {
    MapNop(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.MERGE_LORA_WEIGHT) {
    MapLoRACombination(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.QUANTIZE_TO_P8) {
    MapLoRAQuantize(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.ELWISE_ADD) {
    MapAddition(params, memoryMap, mappedParams, opMemoryMaps);
  } else if (params.MAXPOOL) {
    MapMaxpool(params, memoryMap, mappedParams, opMemoryMaps);
  } else {
    MapMatrixOp(params, memoryMap, mappedParams, opMemoryMaps);
  }
}
