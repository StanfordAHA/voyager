#ifndef MOBILEBERT_PARAMS
#define MOBILEBERT_PARAMS

const SimplifiedParams bottleneck_input_dense_params = {
    66432,                                       // INPUT_OFFSET
    0,                                           // WEIGHT_OFFSET
    131968,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    65536,                                       // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15074192,                                    // WEIGHT_GRADIENT_OFFSET
    15139728,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_input_LayerNorm_params = {
    131968,                                      // INPUT_OFFSET
    65792,                                       // WEIGHT_OFFSET
    181120,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    66048,                                       // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15139984,                                    // WEIGHT_GRADIENT_OFFSET
    15140240,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_dense_params = {
    66432,                                       // INPUT_OFFSET
    66304,                                       // WEIGHT_OFFSET
    148352,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    131840,                                      // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15140496,                                    // WEIGHT_GRADIENT_OFFSET
    15206032,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams bottleneck_attention_LayerNorm_params = {
    148352,                                      // INPUT_OFFSET
    132096,                                      // WEIGHT_OFFSET
    197504,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    132352,                                      // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15206288,                                    // WEIGHT_GRADIENT_OFFSET
    15206544,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_query_layer_params = {
    197504,                                     // INPUT_OFFSET
    132608,                                     // WEIGHT_OFFSET
    213888,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    true,                                       // bias
    148992,                                     // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    true,                                       // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15206800,                                   // WEIGHT_GRADIENT_OFFSET
    15223184,                                   // BIAS_GRADIENT_OFFSET
    -0.02,                                      // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_key_layer_params = {
    197504,                                     // INPUT_OFFSET
    149248,                                     // WEIGHT_OFFSET
    230272,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    true,                                       // bias
    165632,                                     // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    true,                                       // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15223440,                                   // WEIGHT_GRADIENT_OFFSET
    15239824,                                   // BIAS_GRADIENT_OFFSET
    -0.02,                                      // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_scores_0_params = {
    213888,                                     // INPUT_OFFSET
    230272,                                     // WEIGHT_OFFSET
    246656,                                     // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    true,                                       // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15304464,                                   // WEIGHT_GRADIENT_OFFSET
    15074192,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_scores_1_params = {
    217984,                                     // INPUT_OFFSET
    234368,                                     // WEIGHT_OFFSET
    263040,                                     // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    true,                                       // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15308560,                                   // WEIGHT_GRADIENT_OFFSET
    15074192,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_scores_2_params = {
    222080,                                     // INPUT_OFFSET
    238464,                                     // WEIGHT_OFFSET
    279424,                                     // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    true,                                       // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15312656,                                   // WEIGHT_GRADIENT_OFFSET
    15074192,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_scores_3_params = {
    226176,                                     // INPUT_OFFSET
    242560,                                     // WEIGHT_OFFSET
    295808,                                     // OUTPUT_OFFSET
    1,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 2, 1, 1, 1}, {2, 4, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    true,                                       // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15316752,                                   // WEIGHT_GRADIENT_OFFSET
    15074192,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_probs_0_params = {
    246656,                                        // INPUT_OFFSET
    66304,                                         // WEIGHT_OFFSET
    312192,                                        // OUTPUT_OFFSET
    0,                                             // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
    {0, 5},                                        // INPUTX
    {1, 4},                                        // INPUTY
    {3, 0},                                        // REDUCTION
    {2, 1},                                        // WEIGHT
    3,                                             // fxIndex
    2,                                             // fyIndex
    {4, 5},                                        // weightReuseIndex
    1,                                             // stride
    false,                                         // replication
    false,                                         // ReLU
    false,                                         // bias
    0,                                             // BIAS_OFFSET
    false,                                         // residual
    66432,                                         // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    true,                                          // SOFTMAX
    true,                                          // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    false,                                         // SOFTMAX_GRAD
    false,                                         // FC_GRAD
    false,                                         // NO_NORM_GRAD
    false,                                         // RELU_GRAD
    false,                                         // BIAS_GRAD
    false,                                         // CROSS_ENTROPY_GRAD
    false,                                         // MSE_GRAD
    false,                                         // BCE_WITH_LOGITS_GRAD
    false,                                         // INPUT_TRANSPOSE
    false,                                         // CONCAT_INPUT
    false,                                         // CONCAT_WEIGHT
    false,                                         // SPLIT_OUTPUT
    false,                                         // GRAD_CLIPPING
    false,                                         // WEIGHT_SPLITTING
    15140624,                                      // WEIGHT_GRADIENT_OFFSET
    15074192,                                      // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_probs_1_params = {
    263040,                                        // INPUT_OFFSET
    66304,                                         // WEIGHT_OFFSET
    328576,                                        // OUTPUT_OFFSET
    0,                                             // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
    {0, 5},                                        // INPUTX
    {1, 4},                                        // INPUTY
    {3, 0},                                        // REDUCTION
    {2, 1},                                        // WEIGHT
    3,                                             // fxIndex
    2,                                             // fyIndex
    {4, 5},                                        // weightReuseIndex
    1,                                             // stride
    false,                                         // replication
    false,                                         // ReLU
    false,                                         // bias
    0,                                             // BIAS_OFFSET
    false,                                         // residual
    66432,                                         // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    true,                                          // SOFTMAX
    true,                                          // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    false,                                         // SOFTMAX_GRAD
    false,                                         // FC_GRAD
    false,                                         // NO_NORM_GRAD
    false,                                         // RELU_GRAD
    false,                                         // BIAS_GRAD
    false,                                         // CROSS_ENTROPY_GRAD
    false,                                         // MSE_GRAD
    false,                                         // BCE_WITH_LOGITS_GRAD
    false,                                         // INPUT_TRANSPOSE
    false,                                         // CONCAT_INPUT
    false,                                         // CONCAT_WEIGHT
    false,                                         // SPLIT_OUTPUT
    false,                                         // GRAD_CLIPPING
    false,                                         // WEIGHT_SPLITTING
    15140624,                                      // WEIGHT_GRADIENT_OFFSET
    15074192,                                      // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_probs_2_params = {
    279424,                                        // INPUT_OFFSET
    66304,                                         // WEIGHT_OFFSET
    344960,                                        // OUTPUT_OFFSET
    0,                                             // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
    {0, 5},                                        // INPUTX
    {1, 4},                                        // INPUTY
    {3, 0},                                        // REDUCTION
    {2, 1},                                        // WEIGHT
    3,                                             // fxIndex
    2,                                             // fyIndex
    {4, 5},                                        // weightReuseIndex
    1,                                             // stride
    false,                                         // replication
    false,                                         // ReLU
    false,                                         // bias
    0,                                             // BIAS_OFFSET
    false,                                         // residual
    66432,                                         // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    true,                                          // SOFTMAX
    true,                                          // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    false,                                         // SOFTMAX_GRAD
    false,                                         // FC_GRAD
    false,                                         // NO_NORM_GRAD
    false,                                         // RELU_GRAD
    false,                                         // BIAS_GRAD
    false,                                         // CROSS_ENTROPY_GRAD
    false,                                         // MSE_GRAD
    false,                                         // BCE_WITH_LOGITS_GRAD
    false,                                         // INPUT_TRANSPOSE
    false,                                         // CONCAT_INPUT
    false,                                         // CONCAT_WEIGHT
    false,                                         // SPLIT_OUTPUT
    false,                                         // GRAD_CLIPPING
    false,                                         // WEIGHT_SPLITTING
    15140624,                                      // WEIGHT_GRADIENT_OFFSET
    15074192,                                      // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_attention_probs_3_params = {
    295808,                                        // INPUT_OFFSET
    66304,                                         // WEIGHT_OFFSET
    361344,                                        // OUTPUT_OFFSET
    0,                                             // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 128, 128}},  // LOOPS
    {0, 5},                                        // INPUTX
    {1, 4},                                        // INPUTY
    {3, 0},                                        // REDUCTION
    {2, 1},                                        // WEIGHT
    3,                                             // fxIndex
    2,                                             // fyIndex
    {4, 5},                                        // weightReuseIndex
    1,                                             // stride
    false,                                         // replication
    false,                                         // ReLU
    false,                                         // bias
    0,                                             // BIAS_OFFSET
    false,                                         // residual
    66432,                                         // RESIDUAL_OFFSET
    false,                                         // MAXPOOL
    false,                                         // AVGPOOL
    false,                                         // WEIGHT
    false,                                         // STORE_IN_ACC
    false,                                         // ACC_FROM_ACC
    true,                                          // SOFTMAX
    true,                                          // ATTENTION_MASK
    false,                                         // ATTENTION_SCALING
    false,                                         // FC
    false,                                         // NO_NORM
    false,                                         // SOFTMAX_GRAD
    false,                                         // FC_GRAD
    false,                                         // NO_NORM_GRAD
    false,                                         // RELU_GRAD
    false,                                         // BIAS_GRAD
    false,                                         // CROSS_ENTROPY_GRAD
    false,                                         // MSE_GRAD
    false,                                         // BCE_WITH_LOGITS_GRAD
    false,                                         // INPUT_TRANSPOSE
    false,                                         // CONCAT_INPUT
    false,                                         // CONCAT_WEIGHT
    false,                                         // SPLIT_OUTPUT
    false,                                         // GRAD_CLIPPING
    false,                                         // WEIGHT_SPLITTING
    15140624,                                      // WEIGHT_GRADIENT_OFFSET
    15074192,                                      // BIAS_GRADIENT_OFFSET
    0,                                             // learningRate
    false,                                         // ACC_T_INPUT
    false,                                         // ACC_T_WEIGHT
    false,                                         // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_value_layer_params = {
    66432,                                       // INPUT_OFFSET
    165888,                                      // WEIGHT_OFFSET
    164736,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    231424,                                      // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    true,                                        // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15240080,                                    // WEIGHT_GRADIENT_OFFSET
    15305616,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_context_layer_0_params = {
    312192,                                     // INPUT_OFFSET
    164736,                                     // WEIGHT_OFFSET
    377728,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15238928,                                   // WEIGHT_GRADIENT_OFFSET
    15074192,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_context_layer_1_params = {
    328576,                                     // INPUT_OFFSET
    168832,                                     // WEIGHT_OFFSET
    381824,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15243024,                                   // WEIGHT_GRADIENT_OFFSET
    15074192,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_context_layer_2_params = {
    344960,                                     // INPUT_OFFSET
    172928,                                     // WEIGHT_OFFSET
    385920,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15247120,                                   // WEIGHT_GRADIENT_OFFSET
    15074192,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_self_context_layer_3_params = {
    361344,                                     // INPUT_OFFSET
    177024,                                     // WEIGHT_OFFSET
    390016,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 2, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    false,                                      // bias
    0,                                          // BIAS_OFFSET
    false,                                      // residual
    66432,                                      // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    false,                                      // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15251216,                                   // WEIGHT_GRADIENT_OFFSET
    15074192,                                   // BIAS_GRADIENT_OFFSET
    0,                                          // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_output_dense_params = {
    377728,                                     // INPUT_OFFSET
    231680,                                     // WEIGHT_OFFSET
    394112,                                     // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    true,                                       // bias
    248064,                                     // BIAS_OFFSET
    true,                                       // residual
    181120,                                     // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    false,                                      // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    true,                                       // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    15305872,                                   // WEIGHT_GRADIENT_OFFSET
    15322256,                                   // BIAS_GRADIENT_OFFSET
    -0.02,                                      // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

const SimplifiedParams attention_output_LayerNorm_params = {
    394112,                                      // INPUT_OFFSET
    248320,                                      // WEIGHT_OFFSET
    410496,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    248576,                                      // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15322512,                                    // WEIGHT_GRADIENT_OFFSET
    15322768,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_intermediate_dense_params = {
    410496,                                      // INPUT_OFFSET
    248832,                                      // WEIGHT_OFFSET
    426880,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // ReLU
    true,                                        // bias
    314368,                                      // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15323024,                                    // WEIGHT_GRADIENT_OFFSET
    15388560,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_output_dense_params = {
    426880,                                      // INPUT_OFFSET
    315392,                                      // WEIGHT_OFFSET
    492416,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    380928,                                      // BIAS_OFFSET
    true,                                        // residual
    410496,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15389584,                                    // WEIGHT_GRADIENT_OFFSET
    15455120,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams ffn_0_output_LayerNorm_params = {
    492416,                                      // INPUT_OFFSET
    381184,                                      // WEIGHT_OFFSET
    508800,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    381440,                                      // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15455376,                                    // WEIGHT_GRADIENT_OFFSET
    15455632,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams intermediate_dense_params = {
    508800,                                      // INPUT_OFFSET
    381696,                                      // WEIGHT_OFFSET
    525184,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    true,                                        // ReLU
    true,                                        // bias
    447232,                                      // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15455888,                                    // WEIGHT_GRADIENT_OFFSET
    15521424,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams output_dense_params = {
    525184,                                      // INPUT_OFFSET
    448256,                                      // WEIGHT_OFFSET
    590720,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 8, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    513792,                                      // BIAS_OFFSET
    true,                                        // residual
    508800,                                      // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15522448,                                    // WEIGHT_GRADIENT_OFFSET
    15587984,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams output_LayerNorm_params = {
    590720,                                      // INPUT_OFFSET
    514048,                                      // WEIGHT_OFFSET
    607104,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {8, 8, 1, 1, 1, 128}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    514304,                                      // BIAS_OFFSET
    false,                                       // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    true,                                        // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15588240,                                    // WEIGHT_GRADIENT_OFFSET
    15588496,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams output_bottleneck_dense_params = {
    607104,                                      // INPUT_OFFSET
    514560,                                      // WEIGHT_OFFSET
    623488,                                      // OUTPUT_OFFSET
    0,                                           // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {8, 32, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                      // INPUTX
    {1, 4},                                      // INPUTY
    {3, 0},                                      // REDUCTION
    {2, 1},                                      // WEIGHT
    3,                                           // fxIndex
    2,                                           // fyIndex
    {4, 5},                                      // weightReuseIndex
    1,                                           // stride
    false,                                       // replication
    false,                                       // ReLU
    true,                                        // bias
    580096,                                      // BIAS_OFFSET
    true,                                        // residual
    66432,                                       // RESIDUAL_OFFSET
    false,                                       // MAXPOOL
    false,                                       // AVGPOOL
    true,                                        // WEIGHT
    false,                                       // STORE_IN_ACC
    false,                                       // ACC_FROM_ACC
    false,                                       // SOFTMAX
    false,                                       // ATTENTION_MASK
    false,                                       // ATTENTION_SCALING
    false,                                       // FC
    false,                                       // NO_NORM
    false,                                       // SOFTMAX_GRAD
    false,                                       // FC_GRAD
    false,                                       // NO_NORM_GRAD
    false,                                       // RELU_GRAD
    false,                                       // BIAS_GRAD
    false,                                       // CROSS_ENTROPY_GRAD
    false,                                       // MSE_GRAD
    false,                                       // BCE_WITH_LOGITS_GRAD
    false,                                       // INPUT_TRANSPOSE
    false,                                       // CONCAT_INPUT
    false,                                       // CONCAT_WEIGHT
    false,                                       // SPLIT_OUTPUT
    false,                                       // GRAD_CLIPPING
    false,                                       // WEIGHT_SPLITTING
    15588752,                                    // WEIGHT_GRADIENT_OFFSET
    15654288,                                    // BIAS_GRADIENT_OFFSET
    -0.02,                                       // learningRate
    false,                                       // ACC_T_INPUT
    false,                                       // ACC_T_WEIGHT
    false,                                       // ACC_T_OUTPUT
};

const SimplifiedParams output_bottleneck_LayerNorm_params = {
    623488,                                       // INPUT_OFFSET
    581120,                                       // WEIGHT_OFFSET
    689024,                                       // OUTPUT_OFFSET
    0,                                            // WEIGHT_TRANSPOSE
    {{4, 1, 1, 1, 1, 1}, {32, 32, 1, 1, 1, 32}},  // LOOPS
    {0, 5},                                       // INPUTX
    {1, 4},                                       // INPUTY
    {3, 0},                                       // REDUCTION
    {2, 1},                                       // WEIGHT
    3,                                            // fxIndex
    2,                                            // fyIndex
    {4, 5},                                       // weightReuseIndex
    1,                                            // stride
    false,                                        // replication
    false,                                        // ReLU
    true,                                         // bias
    582144,                                       // BIAS_OFFSET
    false,                                        // residual
    66432,                                        // RESIDUAL_OFFSET
    false,                                        // MAXPOOL
    false,                                        // AVGPOOL
    true,                                         // WEIGHT
    false,                                        // STORE_IN_ACC
    false,                                        // ACC_FROM_ACC
    false,                                        // SOFTMAX
    false,                                        // ATTENTION_MASK
    false,                                        // ATTENTION_SCALING
    false,                                        // FC
    true,                                         // NO_NORM
    false,                                        // SOFTMAX_GRAD
    false,                                        // FC_GRAD
    false,                                        // NO_NORM_GRAD
    false,                                        // RELU_GRAD
    false,                                        // BIAS_GRAD
    false,                                        // CROSS_ENTROPY_GRAD
    false,                                        // MSE_GRAD
    false,                                        // BCE_WITH_LOGITS_GRAD
    false,                                        // INPUT_TRANSPOSE
    false,                                        // CONCAT_INPUT
    false,                                        // CONCAT_WEIGHT
    false,                                        // SPLIT_OUTPUT
    false,                                        // GRAD_CLIPPING
    false,                                        // WEIGHT_SPLITTING
    15655312,                                     // WEIGHT_GRADIENT_OFFSET
    15656336,                                     // BIAS_GRADIENT_OFFSET
    -0.02,                                        // learningRate
    false,                                        // ACC_T_INPUT
    false,                                        // ACC_T_WEIGHT
    false,                                        // ACC_T_OUTPUT
};

const SimplifiedParams classifier_params = {
    15008640,                                   // INPUT_OFFSET
    13996032,                                   // WEIGHT_OFFSET
    15074176,                                   // OUTPUT_OFFSET
    0,                                          // WEIGHT_TRANSPOSE
    {{1, 1, 1, 1, 1, 1}, {32, 1, 1, 1, 1, 1}},  // LOOPS
    {0, 5},                                     // INPUTX
    {1, 4},                                     // INPUTY
    {3, 0},                                     // REDUCTION
    {2, 1},                                     // WEIGHT
    3,                                          // fxIndex
    2,                                          // fyIndex
    {4, 5},                                     // weightReuseIndex
    1,                                          // stride
    false,                                      // replication
    false,                                      // ReLU
    true,                                       // bias
    14012416,                                   // BIAS_OFFSET
    false,                                      // residual
    14386048,                                   // RESIDUAL_OFFSET
    false,                                      // MAXPOOL
    false,                                      // AVGPOOL
    true,                                       // WEIGHT
    false,                                      // STORE_IN_ACC
    false,                                      // ACC_FROM_ACC
    false,                                      // SOFTMAX
    false,                                      // ATTENTION_MASK
    false,                                      // ATTENTION_SCALING
    true,                                       // FC
    false,                                      // NO_NORM
    false,                                      // SOFTMAX_GRAD
    false,                                      // FC_GRAD
    false,                                      // NO_NORM_GRAD
    false,                                      // RELU_GRAD
    false,                                      // BIAS_GRAD
    false,                                      // CROSS_ENTROPY_GRAD
    false,                                      // MSE_GRAD
    false,                                      // BCE_WITH_LOGITS_GRAD
    false,                                      // INPUT_TRANSPOSE
    false,                                      // CONCAT_INPUT
    false,                                      // CONCAT_WEIGHT
    false,                                      // SPLIT_OUTPUT
    false,                                      // GRAD_CLIPPING
    false,                                      // WEIGHT_SPLITTING
    29070224,                                   // WEIGHT_GRADIENT_OFFSET
    29086608,                                   // BIAS_GRADIENT_OFFSET
    -0.02,                                      // learningRate
    false,                                      // ACC_T_INPUT
    false,                                      // ACC_T_WEIGHT
    false,                                      // ACC_T_OUTPUT
};

#endif
