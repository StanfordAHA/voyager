set block "Accelerator"
set full_block_name "Accelerator"

proc pre_compile {} {
  global IO_DATATYPE ACCUM_DATATYPE DIMENSION DATATYPE
  set MP_IO_DATATYPE $IO_DATATYPE
    if {$DATATYPE == "HYBRID_FP8"} {
      set MP_IO_DATATYPE "F9"
    }
    foreach mapped_block [list "InputController<$IO_DATATYPE, $DIMENSION>" "MatrixProcessor<$MP_IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION, 1024>" "VectorUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>" "WeightController<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION>"] {
      solution design set $mapped_block -mapped
    }
}

proc pre_libraries {} {
  solution library add {[Block] InputController.v1}
  solution library add {[Block] MatrixProcessor.v1}
  solution library add {[Block] VectorUnit.v1}
  solution library add {[Block] WeightController.v1}
}

proc pre_assembly {} {
  global IO_DATATYPE DATATYPE ACCUM_DATATYPE DIMENSION
  set MP_IO_DATATYPE $IO_DATATYPE
  if {$DATATYPE == "HYBRID_FP8"} {
    set MP_IO_DATATYPE "F9"
  }
  set MatrixProcessorBlock "MatrixProcessor<$MP_IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION, 1024>"
  set MatrixProcessorBlock_stripped [string map {" " ""} $MatrixProcessorBlock]

  set InputControllerBlock "InputController<$IO_DATATYPE, $DIMENSION>"
  set InputControllerBlock_stripped [string map {" " ""} $InputControllerBlock]

  set WeightControllerBlock "WeightController<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION, $DIMENSION>"
  set WeightControllerBlock_stripped [string map {" " ""} $WeightControllerBlock]

  set VectorUnitBlock "VectorUnit<$IO_DATATYPE, $ACCUM_DATATYPE, $DIMENSION>"
  set VectorUnitBlock_stripped [string map {" " ""} $VectorUnitBlock]

  directive set /Accelerator/$MatrixProcessorBlock_stripped -MAP_TO_MODULE {[Block] MatrixProcessor.v1}
  directive set /Accelerator/$InputControllerBlock_stripped -MAP_TO_MODULE {[Block] InputController.v1}
  directive set /Accelerator/$WeightControllerBlock_stripped -MAP_TO_MODULE {[Block] WeightController.v1}
  directive set /Accelerator/$VectorUnitBlock_stripped -MAP_TO_MODULE {[Block] VectorUnit.v1}
}

proc pre_architect {} {
  global IO_DATATYPE DIMENSION C_DATA_REP_NAME IO_DATATYPE_WIDTH
  set double_buffer "DoubleBuffer<$IO_DATATYPE,$DIMENSION,1024>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0.value.$C_DATA_REP_NAME -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]
  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1.value.$C_DATA_REP_NAME -WORD_WIDTH [expr $IO_DATATYPE_WIDTH*$DIMENSION]
}