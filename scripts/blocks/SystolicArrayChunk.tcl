set block "SystolicArrayChunk"
set full_block_name "SystolicArrayChunk<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, 4, $DIMENSION>"

proc pre_compile {} {
  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE PE_PSUM_DATATYPE DIMENSION

  solution design set "SystolicArrayRow<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $DIMENSION>" -mapped
}

proc pre_libraries {} {
  solution library add {[Block] SystolicArrayRow.v1}
}

proc pre_assembly {} {
  global full_block_name
  set full_block_name_stripped [string map {" " ""} $full_block_name]

  global PE_INPUT_DATATYPE PE_WEIGHT_DATATYPE PE_PSUM_DATATYPE DIMENSION
  set row_name "SystolicArrayRow<$PE_INPUT_DATATYPE, $PE_WEIGHT_DATATYPE, $PE_PSUM_DATATYPE, $DIMENSION>"
  set row_name_stripped [string map {" " ""} $row_name]

  directive set /$full_block_name_stripped/$row_name_stripped -MAP_TO_MODULE {[Block] SystolicArrayRow.v1}
}