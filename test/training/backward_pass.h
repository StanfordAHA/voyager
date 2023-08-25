#pragma once
#include "memory_plan.h"

void encoder_backward_pass(int encoderLayer) {
  int activationBase = ENCODER_SCRATCH;
  int activationGradientBase = BACKPROP_SCRATCH;
  int weightBase = encoderLayer * ENCODER_WEIGHT_SIZE;
  int loraWeightOffset = encoderLayer * LORA_W_PER_ENC_SIZE;

  // These namings are weird, they actually refer to the layer before

  run_op(OPERATION(output_bottleneck_dense, backward),
         activationGradientBase + (encoderLayer == 20 ? INTERMEDIATE_SIZE : 0),
         weightBase + 8 * INTERMEDIATE_SIZE + 3 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase, 0, 0);

  run_op(OPERATION(output_LayerNorm, backward), activationGradientBase,
         weightBase + 7 * INTERMEDIATE_SIZE + 2 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE, 0, 0);

  for (int ffn = NUM_FFN - 1; ffn >= 0; ffn--) {
    run_op(OPERATION(output_dense, backward),
           activationGradientBase + INTERMEDIATE_SIZE,
           weightBase + 5 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
               3 * INTRA_BOTTLENECK_SIZE + 13 * INTRA_BOTTLENECK_BIAS_SIZE +
               ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                      3 * INTRA_BOTTLENECK_BIAS_SIZE),
           activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           0, 0);

    forward_pass_from_checkpoint(
        encoderLayer, ffn == NUM_FFN - 1 ? FORWARD_PASS_FFN_1_INTERMEDIATE
                                         : FORWARD_PASS_FFN_0_INTERMEDIATE);

    run_op(
        OPERATION(intermediate_dense, backward),
        activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
        weightBase + 4 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
            3 * INTRA_BOTTLENECK_SIZE + 12 * INTRA_BOTTLENECK_BIAS_SIZE +
            ffn * (2 * INTERMEDIATE_SIZE + +INTERMEDIATE_BIAS_SIZE +
                   3 * INTRA_BOTTLENECK_BIAS_SIZE),
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
        0, activationBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE);

    run_op(
        OPERATION(ffn_0_output_LayerNorm, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
        weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
            12 * INTRA_BOTTLENECK_BIAS_SIZE +
            ffn * (2 * INTERMEDIATE_SIZE + INTERMEDIATE_BIAS_SIZE +
                   3 * INTRA_BOTTLENECK_BIAS_SIZE),
        activationGradientBase + INTERMEDIATE_SIZE, 0,
        activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE);
  }

  run_op(OPERATION(attention_output_dense, backward),
         activationGradientBase + INTERMEDIATE_SIZE,
         weightBase + 3 * INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE +
             10 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE, 0,
         0);

  run_op(OPERATION(attention_self_context_layer, backward),
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         weightBase + 3 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             9 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  run_op(OPERATION(bottleneck_input_dense, backward),
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         weightBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE, 0, 0);

  for (int head = 0; head < NUM_HEADS; head++) {
    forward_pass_from_checkpoint(encoderLayer, head == 0   ? FORWARD_PASS_MHA_0
                                               : head == 1 ? FORWARD_PASS_MHA_1
                                               : head == 2
                                                   ? FORWARD_PASS_MHA_2
                                                   : FORWARD_PASS_MHA_3);
    run_op(OPERATION(attention_self_value_layer_0, backward),
           activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
           activationGradientBase + INTERMEDIATE_SIZE +
               2 * INTRA_BOTTLENECK_SIZE + head * ATTENTION_HEAD_SIZE,
           activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
               head * ATTENTION_HEAD_SIZE,
           0, 0);

    run_op(
        OPERATION(attention_self_attention_probs_0, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
        0, 0);

    run_op(
        OPERATION(attention_self_attention_scores_0, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
        0,
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        0, activationBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE);

    run_op(
        OPERATION(attention_self_query_layer_0, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        0, 0);

    // Merged query weight: lora_A x lora_B + query_weight
    // run_op(
    //     OPERATION(attention_self_query_weight, inference),
    //     LORA_W + loraWeightOffset
    //     LORA_W + loraWeightOffset + LORA_WQ_B_SIZE,
    //     activationBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
    //     0,
    //     weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE);

    // TODO: recompute bottleneck attention LayerNorm
    forward_pass_from_checkpoint(encoderLayer,
                                 FORWARD_PASS_BOTTLENECK_ATTENTION);

    std::cerr << "bottleneck attention LayerNorm" << std::endl;
    int output_loc = activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE;
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 128; j++) {
        int offset = i * 128 + j;
        std::cerr << memory->sram[output_loc + offset] << '\t';
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl << std::endl;

    // query_weight gradient
    run_op(OPERATION(attention_self_query_weight, gradient),
           activationBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
           activationGradientBase + INTERMEDIATE_SIZE +
               5 * INTRA_BOTTLENECK_SIZE + head * ATTENTION_HEAD_SIZE,
           activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
           0, 0);

    std::cerr << "query weight gradient" << std::endl;
    output_loc = activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE;
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 128; j++) {
        int offset = i * 128 + j;
        std::cerr << memory->sram[output_loc + offset] << '\t';
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl << std::endl;

    // query lora_A gradient
    run_op(OPERATION(attention_self_query_lora_A, gradient),
           activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
           LORA_W + loraWeightOffset + LORA_WQ_A_SIZE,
           LORA_G + loraWeightOffset, 0, 0);

    std::cerr << "query lora_A gradient" << std::endl;
    output_loc = LORA_G + loraWeightOffset;
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 16; j++) {
        int offset = i * 16 + j;
        std::cerr << memory->sram[output_loc + 2 * offset] << '\t';
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl << std::endl;

     exit(0);

    // query lora_B gradient
    run_op(OPERATION(attention_self_query_lora_B, gradient),
           LORA_W + loraWeightOffset,
           activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
           LORA_G + loraWeightOffset + LORA_WQ_A_SIZE, 0, 0);

    std::cerr << "query lora B gradient" << std::endl;
    for (int i = 0; i < 128; i++) {
      for (int j = 0; j < 16; j++) {
        int offset = i * 16 + j;
        std::cerr << memory->sram[LORA_G + loraWeightOffset + 2 * offset]
                  << '\t';
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl;

    exit(0);

    run_op(
        OPERATION(attention_self_key_layer_0, backward),
        activationGradientBase + INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_SIZE,
        activationBase + INTERMEDIATE_SIZE + head * ATTENTION_HEAD_SIZE,
        activationGradientBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE +
            head * ATTENTION_HEAD_SIZE,
        0, 0);
  }

  run_op(OPERATION(query_to_bottleneck_attention_LayerNorm, backward),
         activationGradientBase + INTERMEDIATE_SIZE + 5 * INTRA_BOTTLENECK_SIZE,
         weightBase + 2 * INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  run_op(
      OPERATION(key_to_bottleneck_attention_LayerNorm, backward),
      activationGradientBase + INTERMEDIATE_SIZE + 6 * INTRA_BOTTLENECK_SIZE,
      weightBase + 2 * INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE +
          7 * INTRA_BOTTLENECK_BIAS_SIZE,
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE, 0,
      activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE);

  run_op(OPERATION(bottleneck_attention_dense, backward),
         activationGradientBase + INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE,
         weightBase + 2 * INTERMEDIATE_SIZE + 4 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         0, 0);

  run_op(OPERATION(shared_attention_input_to_hidden_states, backward),
         activationGradientBase + INTERMEDIATE_SIZE, weightBase,
         activationGradientBase, 0, activationGradientBase);
  run_op(OPERATION(value_to_hidden_states, backward),
         activationGradientBase + INTERMEDIATE_SIZE + INTRA_BOTTLENECK_SIZE,
         weightBase + 2 * INTERMEDIATE_SIZE + 2 * INTRA_BOTTLENECK_SIZE +
             8 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase, 0, activationGradientBase);
  run_op(OPERATION(bottlenecked_hidden_states, backward),
         activationGradientBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_SIZE,
         weightBase + INTERMEDIATE_SIZE + 3 * INTRA_BOTTLENECK_BIAS_SIZE,
         activationGradientBase, 0, activationGradientBase);
}

void full_backward_pass() {
  // cross-entropy gradient
  run_op(OPERATION(classifier, backward), ENCODER_SCRATCH + INTERMEDIATE_SIZE,
         0, BACKPROP_SCRATCH, 0, 0);

  for (int j = 0; j < 16; j++) {
    std::cerr << memory->sram[BACKPROP_SCRATCH + j] << '\t';
  }
  std::cerr << std::endl;

  // classifier weight
  run_op(OPERATION(classifier_weight, gradient), BACKPROP_SCRATCH,
         ENCODER_SCRATCH, CLASSIFIER_G, 0, 0);

  std::cerr << "classifier weight gradient" << std::endl;
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 512; j++) {
      int offset = i * 512 + j;
      std::cerr << memory->sram[CLASSIFIER_G + 2 * offset] << '\t';
    }
    std::cerr << std::endl;
  }

  // classifier bias
  run_op(OPERATION(classifier_bias, gradient), BACKPROP_SCRATCH, 0,
         CLASSIFIER_G + CLASSIFIER_W_SIZE, 0, 0);

  std::cerr << "classifier bias gradient" << std::endl;
  for (int i = 0; i < 16; i++) {
    std::cerr << memory->sram[CLASSIFIER_G + CLASSIFIER_W_SIZE + 2 * i] << '\t';
  }
  std::cerr << std::endl;

  // classifier gradient
  run_op(OPERATION(output_bottleneck_LayerNorm, backward), BACKPROP_SCRATCH,
         (NUM_ENCODER_LAYERS - 1) * ENCODER_WEIGHT_SIZE +
             8 * INTERMEDIATE_SIZE + 5 * INTERMEDIATE_BIAS_SIZE +
             3 * INTRA_BOTTLENECK_SIZE + 18 * INTRA_BOTTLENECK_BIAS_SIZE,
         BACKPROP_SCRATCH + INTERMEDIATE_SIZE, 0, 0);

  // TODO: pad gradient with 0

  for (int encoderLayer = NUM_ENCODER_LAYERS - 1; encoderLayer >= 0;
       encoderLayer--) {
    encoder_backward_pass(encoderLayer);
  }
}