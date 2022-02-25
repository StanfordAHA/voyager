#pragma once

#include <array>
#include <map>
#include <string>

std::map<std::string, SimplifiedParams> mobilebert{
    // (128 x 512) * (512 x 128)
    // (1x128x512) * (1x1x512x128)
    {"inputBottleneck",
     {
         0,                                           // INPUT_OFFSET
         0,                                           // WEIGHT_OFFSET
         131072,                                      // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{4, 1, 2, 1, 1, 1}, {32, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {5, 5},                                      // weightReuseIndex
         1,                                           // stride
         false,                                       // replication
         false,                                       // RELU
         true,                                        // bias
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // NO-NORM
     }},

    // (128 x 128) x (128 x 128)
    {"qkvProjection",
     {
         131072,                                     // INPUT_OFFSET
         65536,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
     }},

    // attention- Q*KT
    // (128 x 32) * (32 x 128)
    {"qkAttention",
     {
         0,                                          // INPUT_OFFSET
         69632,                                      // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
     }},

    // attention- *v
    // (128 x 128) * (128 x 128)
    {"vAttention",
     {
         131072,                                     // INPUT_OFFSET
         73728,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
     }},

    // wo projection
    // (128 x 128) x (128 x 128)
    {"wProjection",
     {
         0,                                          // INPUT_OFFSET
         77824,                                      // WEIGHT_OFFSET
         131072,                                     // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 4, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         false,                                      // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
     }},

    // FFN 1
    // (128 x 128) * (128 x 512)
    {"ffn1",
     {
         131072,                                     // INPUT_OFFSET
         94208,                                      // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         true,                                       // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
     }},

    // FFN 2
    // (128 x 512) x (512 x 128)
    {"ffn2",
     {
         0,                                           // INPUT_OFFSET
         159744,                                      // WEIGHT_OFFSET
         131072,                                      // OUTPUT_OFFSET
         false,                                       // TRANSPOSE
         {{4, 1, 4, 1, 1, 1}, {32, 2, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                      // INPUTX
         {1, 4},                                      // INPUTY
         {3, 0},                                      // REDUCTION
         {2, 1},                                      // WEIGHT
         3,                                           // fxIndex
         2,                                           // fyIndex
         {5, 5},                                      // weightReuseIndex
         1,                                           // stride
         false,                                       // replication
         false,                                       // RELU
         true,                                        // bias
         30 * 1024,                                   // BIAS_OFFSET
         false,                                       // residual
         40 * 1024,                                   // RESIDUAL_OFFSET
         false,                                       // maxpool
         false,                                       // avgpool
         false,                                       // SOFTMAX
         false,                                       // FC
         false,                                       // NO-NORM
     }},

    // output bottleneck
    // (128 x 128) x (128 x 512)
    {"outputBottleneck",
     {
         131072,                                     // INPUT_OFFSET
         225280,                                     // WEIGHT_OFFSET
         0,                                          // OUTPUT_OFFSET
         false,                                      // TRANSPOSE
         {{4, 1, 8, 1, 1, 1}, {8, 4, 1, 1, 1, 32}},  // LOOPS
         {0, 5},                                     // INPUTX
         {1, 4},                                     // INPUTY
         {3, 0},                                     // REDUCTION
         {2, 1},                                     // WEIGHT
         3,                                          // fxIndex
         2,                                          // fyIndex
         {5, 5},                                     // weightReuseIndex
         1,                                          // stride
         false,                                      // replication
         false,                                      // RELU
         true,                                       // bias
         30 * 1024,                                  // BIAS_OFFSET
         false,                                      // residual
         40 * 1024,                                  // RESIDUAL_OFFSET
         false,                                      // maxpool
         false,                                      // avgpool
         false,                                      // SOFTMAX
         false,                                      // FC
         false,                                      // NO-NORM
     }},
};

std::array<std::string, 8> mobilebert_order{
    "inputBottleneck", "qkvProjection", "qkAttention", "vAttention",
    "wProjection",     "ffn1",          "ffn2",        "outputBottleneck"};

std::map<std::string, std::string> mobilebertOps{
    {"bottleneck_input_dense", "inputBottleneck"},
    {"bottleneck_attention_dense", "inputBottleneck"},
    {"attention_self_query", "qkvProjection"},
    {"attention_self_key", "qkvProjection"},
    {"attention_self_value", "inputBottleneck"},
    {"attention_self_attention_scores", "qkAttention"},  // need transpose
    // {"attention_self_attention_probs", ""},
    {"attention_self_context_layer", "vAttention"},
    {"attention_output_dense", "wProjection"},
    {"ffn_0_intermediate_dense", "ffn1"},
    {"ffn_0_output_dense", "ffn2"},
    {"ffn_1_intermediate_dense", "ffn1"},
    {"ffn_1_output_dense", "ffn2"},
    {"ffn_2_intermediate_dense", "ffn1"},
    {"ffn_2_output_dense", "ffn2"},
    {"intermediate_dense", "ffn1"},
    {"output_dense", "ffn2"},
    {"output_bottleneck_dense", "outputBottleneck"}};

std::string mobilebertDataDir = "data/mobilebert/";

std::map<std::string, Files> mobilebertFiles{
    {"bottleneck_input_dense",
     {"activations/mobilebert_embeddings",
      "weights/mobilebert_encoder_layer_0_bottleneck_input_dense_weight",
      "weights/mobilebert_encoder_layer_0_bottleneck_input_dense_bias",
      "activations/mobilebert_encoder_layer_0_bottleneck_input_dense"}},
    // {"bottleneck_input_LayerNorm", {}},
    {"bottleneck_attention_dense",
     {"activations/mobilebert_embeddings",
      "weights/mobilebert_encoder_layer_0_bottleneck_attention_dense_weight",
      "weights/mobilebert_encoder_layer_0_bottleneck_attention_dense_bias",
      "activations/mobilebert_encoder_layer_0_bottleneck_attention_dense"}},
    // {"bottleneck_attention_LayerNorm", {}},
    {"attention_self_query",
     {"activations/mobilebert_encoder_layer_0_bottleneck_attention_LayerNorm",
      "weights/mobilebert_encoder_layer_0_attention_self_query_weight",
      "weights/mobilebert_encoder_layer_0_attention_self_query_bias",
      "activations/mobilebert_encoder_layer_0_attention_self_query"}},
    {"attention_self_key",
     {"activations/mobilebert_encoder_layer_0_bottleneck_attention_LayerNorm",
      "weights/mobilebert_encoder_layer_0_attention_self_key_weight",
      "weights/mobilebert_encoder_layer_0_attention_self_key_bias",
      "activations/mobilebert_encoder_layer_0_attention_self_key"}},
    {"attention_self_value",
     {"activations/mobilebert_embeddings",
      "weights/mobilebert_encoder_layer_0_attention_self_value_weight",
      "weights/mobilebert_encoder_layer_0_attention_self_value_bias",
      "activations/mobilebert_encoder_layer_0_attention_self_value"}},

    // TODO: break the following operations into 4 sub-ops.
    {"attention_self_attention_scores",
     {"activations/mobilebert_encoder_layer_0_attention_self_query",
      "activations/mobilebert_encoder_layer_0_attention_self_key", "N/A",
      "activations/"
      "mobilebert_encoder_layer_0_attention_self_attention_scores"}},
    // {"attention_self_attention_probs", {}},
    {"attention_self_context_layer",
     {"activations/mobilebert_encoder_layer_0_attention_self_attention_probs",
      "activations/mobilebert_encoder_layer_0_attention_self_value", "N/A",
      "activations/mobilebert_encoder_layer_0_attention_self_context_layer"}},
    {"attention_output_dense",
     {"activations/mobilebert_encoder_layer_0_attention_self_context_layer",
      "weights/mobilebert_encoder_layer_0_attention_output_dense_weight",
      "weights/mobilebert_encoder_layer_0_attention_output_dense_bias",
      "activations/mobilebert_encoder_layer_0_attention_output_dense"}},
    // {"attention_output_LayerNorm", {}},
    // TODO: end

    {"ffn_0_intermediate_dense",
     {"activations/mobilebert_encoder_layer_0_attention_output_LayerNorm",
      "weights/mobilebert_encoder_layer_0_ffn_0_intermediate_dense_weight",
      "weights/mobilebert_encoder_layer_0_ffn_0_intermediate_dense_bias",
      "activations/mobilebert_encoder_layer_0_ffn_0_intermediate_dense"}},
    {"ffn_0_output_dense",
     {"activations/mobilebert_encoder_layer_0_ffn_0_intermediate_dense",
      "weights/mobilebert_encoder_layer_0_ffn_0_output_dense_weight",
      "weights/mobilebert_encoder_layer_0_ffn_0_output_dense_bias",
      "activations/mobilebert_encoder_layer_0_ffn_0_output_dense"}},
    // {"ffn_0_output_LayerNorm", {}},
    {"ffn_1_intermediate_dense",
     {"activations/mobilebert_encoder_layer_0_ffn_0_output_LayerNorm",
      "weights/mobilebert_encoder_layer_0_ffn_1_intermediate_dense_weight",
      "weights/mobilebert_encoder_layer_0_ffn_1_intermediate_dense_bias",
      "activations/mobilebert_encoder_layer_0_ffn_1_intermediate_dense"}},
    {"ffn_1_output_dense",
     {"activations/mobilebert_encoder_layer_0_ffn_1_intermediate_dense",
      "weights/mobilebert_encoder_layer_0_ffn_1_output_dense_weight",
      "weights/mobilebert_encoder_layer_0_ffn_1_output_dense_bias",
      "activations/mobilebert_encoder_layer_0_ffn_1_output_dense"}},
    // {"ffn_1_output_LayerNorm", {}},
    {"ffn_2_intermediate_dense",
     {"activations/mobilebert_encoder_layer_0_ffn_1_output_LayerNorm",
      "weights/mobilebert_encoder_layer_0_ffn_2_intermediate_dense_weight",
      "weights/mobilebert_encoder_layer_0_ffn_2_intermediate_dense_bias",
      "activations/mobilebert_encoder_layer_0_ffn_2_intermediate_dense"}},
    {"ffn_2_output_dense",
     {"activations/mobilebert_encoder_layer_0_ffn_2_intermediate_dense",
      "weights/mobilebert_encoder_layer_0_ffn_2_output_dense_weight",
      "weights/mobilebert_encoder_layer_0_ffn_2_output_dense_bias",
      "activations/mobilebert_encoder_layer_0_ffn_2_output_dense"}},
    // {"ffn_2_output_LayerNorm", {}},
    {"intermediate_dense",
     {"activations/mobilebert_encoder_layer_0_ffn_2_output_LayerNorm",
      "weights/mobilebert_encoder_layer_0_intermediate_dense_weight",
      "weights/mobilebert_encoder_layer_0_intermediate_dense_bias",
      "activations/mobilebert_encoder_layer_0_intermediate_dense"}},
    {"output_dense",
     {"activations/mobilebert_encoder_layer_0_intermediate_dense",
      "weights/mobilebert_encoder_layer_0_output_dense_weight",
      "weights/mobilebert_encoder_layer_0_output_dense_bias",
      "activations/mobilebert_encoder_layer_0_output_dense"}},
    // {"output_LayerNorm", {}},
    {"output_bottleneck_dense",
     {"activations/mobilebert_encoder_layer_0_output_LayerNorm",
      "weights/mobilebert_encoder_layer_0_output_bottleneck_dense_weight",
      "weights/mobilebert_encoder_layer_0_output_bottleneck_dense_bias",
      "activations/mobilebert_encoder_layer_0_output_bottleneck_dense"}},
    // {"output_bottleneck_LayerNorm", {}},
};

std::map<std::string, MemoryMap> mobilebertMemoryMap{
    {"bottleneck_input_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"bottleneck_attention_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_query", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_key", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_value", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_scores", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_attention_probs", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_self_context_layer", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"attention_output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_0_intermediate_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_0_output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_1_intermediate_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_1_output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_2_intermediate_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"ffn_2_output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"intermediate_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"output_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}},
    {"output_bottleneck_dense", {SRAM, RRAM, RRAM, SRAM, SRAM}}};
