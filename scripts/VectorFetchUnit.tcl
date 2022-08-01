source scripts/architecture.tcl

set block "VectorFetchUnit"
set full_block_name "VectorFetchUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>"

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

go extract
