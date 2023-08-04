#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "test/common/VerificationTypes.h"

#define NO_SYSC
// clang-format off
#include "src/PositTypes.h"
// clang-format on
#include "src/ArchitectureParams.h"
#include "test/common/GoldModel.h"
#include "test/common/SimpleMemoryModel.h"
#include "test/common/Utils.h"
#include "test/mobilebert/MobileBERT.h"

/* Test accuracy of model on a dataset using the Posit gold model.
 */
int main(int argc, char* argv[]) {
  // model name is first argument
  std::string modelName = std::string(argv[1]);
  // path to folder with model parameters and dataset is second argument
  std::string modelPath = std::string(argv[2]);

  // Get vector of workloads to run
  if (modelName != "mobilebert") {
    // TODO: Add other models
    std::cout << "ERROR: Model name not recognized." << std::endl;
  }

  // Get forward pass workloads
  MobileBERT mobilebert("mobilebert", "inference", "");
  std::vector<Workload> workloads = mobilebert.getFullForwardPass();

  validateMemoryMapping(workloads);

  // Create the memory model and load the model parameters
  SimpleMemoryModel<INPUT_DATATYPE> memory(false);

  std::cout << "Loading model parameters..." << std::endl;
  for (Workload& workload : workloads) {
    if (workload.loadWeight) {
      workload.files.weights_file.insert(0, modelPath + "/params/");
      workload.files.bias_file.insert(0, modelPath + "/params/");
      memory.loadModelParams(workload.params, workload.files,
                             workload.memoryMap, true);
    }
  }

  int numSamples = 0;
  int numCorrect = 0;

  // iterate through all folders in modelPath
  // each folder consists of two inputs
  for (const auto& entry : std::filesystem::directory_iterator(modelPath)) {
    if (entry.is_directory()) {
      // get folder name
      std::string folderName = entry.path().filename().string();

      // skip params folder
      if (folderName == "params") {
        continue;
      }

      // load the model inputs for this sample
      Workload firstWorkload = workloads[0];
      firstWorkload.files.inputs_file =
          modelPath + "/" + folderName + "/embedding_output";
      memory.loadModelActivations(firstWorkload.params, firstWorkload.files,
                                  firstWorkload.memoryMap, true);

      // for loading attention mask, we make a custom workload
      Workload dummyAttentionMaskWorkload = mobilebert.getWorkloadsInRange(
          {"attention_self_attention_scores_0"})[0];
      dummyAttentionMaskWorkload.files.inputs_file = "";
      dummyAttentionMaskWorkload.files.weights_file = "";
      dummyAttentionMaskWorkload.files.bias_file =
          modelPath + "/" + folderName + "/attention_mask";

      dummyAttentionMaskWorkload.params.ATTENTION_MASK = true;
      dummyAttentionMaskWorkload.params.RESIDUAL = false;
      dummyAttentionMaskWorkload.params.RELU_GRAD = false;
      dummyAttentionMaskWorkload.params.SOFTMAX_GRAD = false;
      dummyAttentionMaskWorkload.params.WEIGHT_SPLITTING = false;
      dummyAttentionMaskWorkload.memoryMap.bias = SRAM;
      memory.loadModelActivations(dummyAttentionMaskWorkload.params,
                                  dummyAttentionMaskWorkload.files,
                                  dummyAttentionMaskWorkload.memoryMap, true);

      std::cout << "Running inference on " << folderName << "..." << std::endl;
      // run every workload in the forward pass
      for (Workload& workload : workloads) {
        run_custom_posit_gold_model(
            workload.params, memory.sram + workload.params.INPUT_OFFSET,
            (workload.memoryMap.weights ? memory.rram : memory.sram) +
                workload.params.WEIGHT_OFFSET,
            memory.sram + workload.params.OUTPUT_OFFSET,
            (workload.memoryMap.bias ? memory.rram : memory.sram) +
                workload.params.BIAS_OFFSET,
            memory.sram + workload.params.RESIDUAL_OFFSET,
            memory.sram + workload.params.WEIGHT_RESIDUAL_OFFSET);
      }

      // get final output
      float outputClassification[2];
      outputClassification[0] =
          memory.sram[workloads.back().params.OUTPUT_OFFSET];
      outputClassification[1] =
          memory.sram[workloads.back().params.OUTPUT_OFFSET + 1];

      std::cout << "Output: " << outputClassification[0] << ", "
                << outputClassification[1] << std::endl;

      numSamples++;

      // get ground truth from folder name, it's the last character
      int groundTruth = folderName.back() - '0';

      // check if output matches ground truth
      if (outputClassification[groundTruth] >
          outputClassification[1 - groundTruth]) {
        numCorrect++;
      }

      std::cout << "Accuracy: " << numCorrect << "/" << numSamples << " ("
                << float(numCorrect) / numSamples * 100 << "%)" << std::endl;
    }
  }
}
