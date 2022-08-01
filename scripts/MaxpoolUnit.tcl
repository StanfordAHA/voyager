source scripts/architecture.tcl

set block "MaxpoolUnit"
set full_block_name "MaxpoolUnit<$ACCUM_DATATYPE, $IO_DATATYPE, $DIMENSION>"

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

directive set /MaxpoolUnit<P16,P8,16>/run/while:maxpool_comparator.value.bits:rsc -MAP_TO_MODULE {[Register]}

go extract
