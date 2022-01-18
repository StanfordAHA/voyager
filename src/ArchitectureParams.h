#define POSIT

#ifdef POSIT
#define INPUT_DATATYPE Posit
#define WEIGHT_DATATYPE Posit
#define ACCUM_DATATYPE PositFP
#define OUTPUT_DATATYPE Posit
#else
#define INPUT_DATATYPE ac_int<8, true>
#define WEIGHT_DATATYPE ac_int<8, true>
#define ACCUM_DATATYPE ac_int<32, true>
#define OUTPUT_DATATYPE ac_int<8, true>
#endif

#define DIMENSION 16
#define INPUT_BUFFER_SIZE 1024
#define WEIGHT_BUFFER_SIZE 1024
#define ACCUMULATION_BUFFER_SIZE 1024
