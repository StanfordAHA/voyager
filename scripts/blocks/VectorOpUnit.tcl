set block "Vectoreduce_opUnit"
set full_block_name "Vectoreduce_opUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>"
set clock_multiplier 1.4

proc pre_architect {} {
  global full_block_name VECTOR_DATATYPE ACCUM_BUFFER_DATATYPE
  set vector_op_unit_stripped [string map {" " ""} $full_block_name]
}
