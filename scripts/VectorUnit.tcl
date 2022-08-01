source scripts/architecture.tcl

set block "VectorUnit"
set full_block_name "VectorUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>"

source scripts/common.tcl

solution library add {[Block] MaxpoolUnit.v1}
solution library add {[Block] VectorFetchUnit.v1}
solution library add {[Block] OutputAddressGenerator.v1}
solution library add {[Block] VectorOpUnit.v1}

go libraries

directive set /VectorUnit<P8,P16,16>/MaxpoolUnit<P8,P16,16> -MAP_TO_MODULE {[Block] MaxpoolUnit.v1}
directive set /VectorUnit<P8,P16,16>/VectorFetchUnit<P8,P16,16> -MAP_TO_MODULE {[Block] VectorFetchUnit.v1}
directive set /VectorUnit<P8,P16,16>/OutputAddressGenerator<16> -MAP_TO_MODULE {[Block] OutputAddressGenerator.v1}
directive set /VectorUnit<P8,P16,16>/VectorOpUnit<P8,P16,16> -MAP_TO_MODULE {[Block] VectorOpUnit.v1}

directive set -CLOCKS $clocks

go assembly

go extract
