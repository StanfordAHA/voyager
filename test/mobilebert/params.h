// (128 x 512) * (512 x 128)
const Params inputBottleneck = {
    32,                        // M0
    2,                         // P1
    32,                        // N1
    4,                         // M1
    4,                         // P2
    0,                         // INPUT_OFFSET
    1024 * 1024,               // WEIGHT_OFFSET
    2 * 1024 * 1024,           // OUTPUT_OFFSET
    false,                     // SOFTMAX
    1,                         // SCALE
    false,                     // TRANSPOSE
    0,                         // VECTOR_OFFSET
    false,                     // VEC_OP
    false,                     // VEC_SUB
    false,                     // VEC_SQUARE
    false,                     // VEC_REDUCE
    true,                      // CONST_SCALE
    0,                         // VEC_SCALE_OFFSET
    0,                         // VEC_SUB_OFFSET
    false,                     // RELU
    {{4, 4, 1}, {32, 2, 32}},  // LOOPS
    {1, 2},                    // INPUT
    {2, 0},                    // REDUCTION
    {0, 1}                     // WEIGHT
};

// (128 x 128) x (128 x 32)
const Params qkvProjection = {
    32,                       // M0
    2,                        // P1
    8,                        // N1
    4,                        // M1
    1,                        // P2
    0,                        // INPUT_OFFSET
    1024 * 1024,              // WEIGHT_OFFSET
    2 * 1024 * 1024,          // OUTPUT_OFFSET
    false,                    // SOFTMAX
    1,                        // SCALE
    false,                    // TRANSPOSE
    0,                        // VECTOR_OFFSET
    false,                    // VEC_OP
    false,                    // VEC_SUB
    false,                    // VEC_SQUARE
    false,                    // VEC_REDUCE
    true,                     // CONST_SCALE
    0,                        // VEC_SCALE_OFFSET
    0,                        // VEC_SUB_OFFSET
    false,                    // RELU
    {{1, 4, 1}, {8, 2, 32}},  // LOOPS
    {1, 2},                   // INPUT
    {2, 0},                   // REDUCTION
    {0, 1}                    // WEIGHT
};

// attention- Q*KT
// (128 x 32) * (32 x 128)
const Params qkAttention = {
    32,                       // M0
    2,                        // P1
    2,                        // N1
    4,                        // M1
    4,                        // P2
    0,                        // INPUT_OFFSET
    1024 * 1024,              // WEIGHT_OFFSET
    2 * 1024 * 1024,          // OUTPUT_OFFSET
    false,                    // SOFTMAX
    1,                        // SCALE
    false,                    // TRANSPOSE
    0,                        // VECTOR_OFFSET
    false,                    // VEC_OP
    false,                    // VEC_SUB
    false,                    // VEC_SQUARE
    false,                    // VEC_REDUCE
    true,                     // CONST_SCALE
    0,                        // VEC_SCALE_OFFSET
    0,                        // VEC_SUB_OFFSET
    false,                    // RELU
    {{4, 4, 1}, {2, 2, 32}},  // LOOPS
    {1, 2},                   // INPUT
    {2, 0},                   // REDUCTION
    {0, 1}                    // WEIGHT
};

// attention- *v
// (128 x 128) * (128 x 32)
const Params vAttention = {
    32,                       // M0
    2,                        // P1
    8,                        // N1
    4,                        // M1
    1,                        // P2
    0,                        // INPUT_OFFSET
    1024 * 1024,              // WEIGHT_OFFSET
    2 * 1024 * 1024,          // OUTPUT_OFFSET
    false,                    // SOFTMAX
    1,                        // SCALE
    false,                    // TRANSPOSE
    0,                        // VECTOR_OFFSET
    false,                    // VEC_OP
    false,                    // VEC_SUB
    false,                    // VEC_SQUARE
    false,                    // VEC_REDUCE
    true,                     // CONST_SCALE
    0,                        // VEC_SCALE_OFFSET
    0,                        // VEC_SUB_OFFSET
    false,                    // RELU
    {{1, 4, 1}, {8, 2, 32}},  // LOOPS
    {1, 2},                   // INPUT
    {2, 0},                   // REDUCTION
    {0, 1}                    // WEIGHT
};

// wo projection
// (128 x 128) x (128 x 128)
const Params wProjection = {
    32,                       // M0
    2,                        // P1
    8,                        // N1
    4,                        // M1
    4,                        // P2
    0,                        // INPUT_OFFSET
    1024 * 1024,              // WEIGHT_OFFSET
    2 * 1024 * 1024,          // OUTPUT_OFFSET
    false,                    // SOFTMAX
    1,                        // SCALE
    false,                    // TRANSPOSE
    0,                        // VECTOR_OFFSET
    false,                    // VEC_OP
    false,                    // VEC_SUB
    false,                    // VEC_SQUARE
    false,                    // VEC_REDUCE
    true,                     // CONST_SCALE
    0,                        // VEC_SCALE_OFFSET
    0,                        // VEC_SUB_OFFSET
    false,                    // RELU
    {{4, 4, 1}, {8, 2, 32}},  // LOOPS
    {1, 2},                   // INPUT
    {2, 0},                   // REDUCTION
    {0, 1}                    // WEIGHT
};

// FFN 1
// (128 x 128) * (128 x 512)
const Params ffn1 = {
    32,                       // M0
    4,                        // P1
    8,                        // N1
    4,                        // M1
    8,                        // P2
    0,                        // INPUT_OFFSET
    1024 * 1024,              // WEIGHT_OFFSET
    2 * 1024 * 1024,          // OUTPUT_OFFSET
    false,                    // SOFTMAX
    1,                        // SCALE
    false,                    // TRANSPOSE
    0,                        // VECTOR_OFFSET
    false,                    // VEC_OP
    false,                    // VEC_SUB
    false,                    // VEC_SQUARE
    false,                    // VEC_REDUCE
    true,                     // CONST_SCALE
    0,                        // VEC_SCALE_OFFSET
    0,                        // VEC_SUB_OFFSET
    false,                    // RELU
    {{8, 4, 1}, {8, 4, 32}},  // LOOPS
    {1, 2},                   // INPUT
    {2, 0},                   // REDUCTION
    {0, 1}                    // WEIGHT
};

// FFN 2
// (128 x 512) x (512 x 128)
const Params ffn2 = {
    32,                        // M0
    2,                         // P1
    32,                        // N1
    4,                         // M1
    4,                         // P2
    0,                         // INPUT_OFFSET
    1024 * 1024,               // WEIGHT_OFFSET
    2 * 1024 * 1024,           // OUTPUT_OFFSET
    false,                     // SOFTMAX
    1,                         // SCALE
    false,                     // TRANSPOSE
    0,                         // VECTOR_OFFSET
    false,                     // VEC_OP
    false,                     // VEC_SUB
    false,                     // VEC_SQUARE
    false,                     // VEC_REDUCE
    true,                      // CONST_SCALE
    0,                         // VEC_SCALE_OFFSET
    0,                         // VEC_SUB_OFFSET
    false,                     // RELU
    {{4, 4, 1}, {32, 2, 32}},  // LOOPS
    {1, 2},                    // INPUT
    {2, 0},                    // REDUCTION
    {0, 1}                     // WEIGHT
};

// output bottleneck
// (128 x 128) x (128 x 512)
const Params outputBottleneck = {
    32,                       // M0
    4,                        // P1
    8,                        // N1
    4,                        // M1
    8,                        // P2
    0,                        // INPUT_OFFSET
    1024 * 1024,              // WEIGHT_OFFSET
    2 * 1024 * 1024,          // OUTPUT_OFFSET
    false,                    // SOFTMAX
    1,                        // SCALE
    false,                    // TRANSPOSE
    0,                        // VECTOR_OFFSET
    false,                    // VEC_OP
    false,                    // VEC_SUB
    false,                    // VEC_SQUARE
    false,                    // VEC_REDUCE
    true,                     // CONST_SCALE
    0,                        // VEC_SCALE_OFFSET
    0,                        // VEC_SUB_OFFSET
    false,                    // RELU
    {{8, 4, 1}, {8, 4, 32}},  // LOOPS
    {1, 2},                   // INPUT
    {2, 0},                   // REDUCTION
    {0, 1}                    // WEIGHT
};
