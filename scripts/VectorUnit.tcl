source scripts/architecture.tcl

set block "VectorUnit"
set full_block_name "VectorUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>"
set full_block_name_stripped [string map {" " ""} $full_block_name]

set maxpool_name "MaxpoolUnit<$ACCUM_DATATYPE,$IO_DATATYPE,$DIMENSION>"
set maxpool_name_stripped [string map {" " ""} $maxpool_name]
set vector_fetch_name "VectorFetchUnit<$IO_DATATYPE,$ACCUM_DATATYPE,$DIMENSION>"
set vector_fetch_name_stripped [string map {" " ""} $vector_fetch_name]
set output_address_name "OutputAddressGenerator<$DIMENSION>"
set output_address_name_stripped [string map {" " ""} $output_address_name]
set vector_op_name "VectorOpUnit<$IO_DATATYPE,$ACCUM_DATATYPE,$DIMENSION>"
set vector_op_name_stripped [string map {" " ""} $vector_op_name]

source scripts/common.tcl

solution library add {[Block] MaxpoolUnit.v1}
solution library add {[Block] VectorFetchUnit.v1}
solution library add {[Block] OutputAddressGenerator.v1}
solution library add {[Block] VectorOpUnit.v1}

go libraries

directive set /$full_block_name_stripped/$maxpool_name_stripped -MAP_TO_MODULE {[Block] MaxpoolUnit.v1}
directive set /$full_block_name_stripped/$vector_fetch_name_stripped -MAP_TO_MODULE {[Block] VectorFetchUnit.v1}
directive set /$full_block_name_stripped/$output_address_name_stripped -MAP_TO_MODULE {[Block] OutputAddressGenerator.v1}
directive set /$full_block_name_stripped/$vector_op_name_stripped -MAP_TO_MODULE {[Block] VectorOpUnit.v1}

directive set -CLOCKS $clocks

go assembly

go extract
