set block "ProcessingElement"
set full_block_name "ProcessingElement<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $ACCUM_DATATYPE>"
set clock_multiplier 1.3

proc pre_extract {} {
   cycle set input_in.Pop() -from psum_in.Pop() -equal 0
}

#proc pre_assembly {} {
#  directive set -CHAN_IO_PROTOCOL coupled
#}
#proc pre_extract {} {
#  cycle set input_out.Push() -from input_in.Pop() -equal 1
#}
