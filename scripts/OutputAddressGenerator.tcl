source scripts/architecture.tcl

set block "OutputAddressGenerator"
set full_block_name "OutputAddressGenerator<$DIMENSION>"

source scripts/common.tcl

go libraries

directive set -CLOCKS $clocks

go assembly

go extract
