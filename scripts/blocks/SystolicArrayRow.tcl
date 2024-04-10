set block "SystolicArrayRow"
set full_block_name "SystolicArrayRow<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $DIMENSION>"


proc pre_libraries {} {
  solution library add {[Block] ProcessingElement.v1}
}

proc pre_assembly {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE PE_PSUM_DATATYPE
  set pe_name "ProcessingElement<$PE_INPUT_DATATYPE,$PE_WEIGHT_DATATYPE,$PE_PSUM_DATATYPE>"
  set pe_name_stripped [string map {" " ""} $pe_name]

  directive set /$full_block_name_stripped/$pe_name_stripped -MAP_TO_MODULE {[Block] ProcessingElement.v1}
}