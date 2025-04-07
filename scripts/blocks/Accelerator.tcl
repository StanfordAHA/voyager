set block "Accelerator"
set full_block_name "Accelerator"

proc pre_compile {} {
  global IO_DATATYPE ACCUM_DATATYPE VECTOR_DATATYPE IC_DIMENSION OC_DIMENSION DATATYPE ACCUM_BUFFER_DATATYPE SUPPORT_MX SCALE_DATATYPE ACCUM_BUFFER_SIZE
  foreach mapped_block [list "InputController<$IO_DATATYPE, $IC_DIMENSION>" "MatrixProcessor<$IO_DATATYPE, $IO_DATATYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>" "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>" "WeightController<$IO_DATATYPE, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"] {
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
  global IO_DATATYPE DATATYPE ACCUM_DATATYPE VECTOR_DATATYPE IC_DIMENSION OC_DIMENSION ACCUM_BUFFER_DATATYPE SCALE_DATATYPE SUPPORT_MX ACCUM_BUFFER_SIZE
  set MatrixProcessorBlock "MatrixProcessor<$IO_DATATYPE, $IO_DATATYPE, $ACCUM_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $IC_DIMENSION, $OC_DIMENSION, $ACCUM_BUFFER_SIZE>"
  set MatrixProcessorBlock_stripped [string map {" " ""} $MatrixProcessorBlock]

  set InputControllerBlock "InputController<$IO_DATATYPE, $IC_DIMENSION>"
  set InputControllerBlock_stripped [string map {" " ""} $InputControllerBlock]

  set WeightControllerBlock "WeightController<$IO_DATATYPE, $ACCUM_BUFFER_DATATYPE, $IC_DIMENSION, $OC_DIMENSION>"
  set WeightControllerBlock_stripped [string map {" " ""} $WeightControllerBlock]

  set VectorUnitBlock "VectorUnit<$VECTOR_DATATYPE, $ACCUM_BUFFER_DATATYPE, $SCALE_DATATYPE, $OC_DIMENSION>"
  set VectorUnitBlock_stripped [string map {" " ""} $VectorUnitBlock]

  directive set /Accelerator/$MatrixProcessorBlock_stripped -MAP_TO_MODULE {[Block] MatrixProcessor.v1}
  directive set /Accelerator/$InputControllerBlock_stripped -MAP_TO_MODULE {[Block] InputController.v1}
  directive set /Accelerator/$WeightControllerBlock_stripped -MAP_TO_MODULE {[Block] WeightController.v1}
  directive set /Accelerator/$VectorUnitBlock_stripped -MAP_TO_MODULE {[Block] VectorUnit.v1}
}

proc pre_architect {} {
  global IO_DATATYPE IC_DIMENSION OC_DIMENSION IO_DATATYPE_WIDTH TECHNOLOGY memories ACCUM_BUFFER_DATATYPE SUPPORT_MX INPUT_BUFFER_SIZE WEIGHT_BUFFER_SIZE IC_PORT_WIDTH OC_PORT_WIDTH
  set double_buffer "DoubleBuffer<$IC_PORT_WIDTH,$INPUT_BUFFER_SIZE>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

  set memory_width [expr $IO_DATATYPE_WIDTH*$IC_DIMENSION]

  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0 -WORD_WIDTH $memory_width
  directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1 -WORD_WIDTH $memory_width

  if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0:rsc -MAP_TO_MODULE [get_memory_name 1 $INPUT_BUFFER_SIZE $memory_width]
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1:rsc -MAP_TO_MODULE [get_memory_name 1 $INPUT_BUFFER_SIZE $memory_width]
  }

  # Weight double buffer
  # When this is exactly the same as the input double buffer, same directives just repeat
  set double_buffer "DoubleBuffer<$OC_PORT_WIDTH,$WEIGHT_BUFFER_SIZE>"
  set double_buffer_stripped [string map {" " ""} $double_buffer]

  set memory_width [expr $IO_DATATYPE_WIDTH*$OC_DIMENSION]

    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0 -WORD_WIDTH $memory_width
    directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1 -WORD_WIDTH $memory_width

    if {$TECHNOLOGY != "generic" && $TECHNOLOGY != "tsmc40"} {
      directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem0Run/mem0Run/mem0:rsc -MAP_TO_MODULE [get_memory_name 1 $WEIGHT_BUFFER_SIZE $memory_width]
      directive set /Accelerator/$double_buffer_stripped/$double_buffer_stripped:mem1Run/mem1Run/mem1:rsc -MAP_TO_MODULE [get_memory_name 1 $WEIGHT_BUFFER_SIZE $memory_width]
    }

  if {$SUPPORT_MX == true} {
    global SCALE_DATATYPE SCALE_DATATYPE_WIDTH

    # Weight scale buffer 
    set WEIGHT_SCALE_BUFFER_SIZE [expr {$WEIGHT_BUFFER_SIZE / 32}] 
    set weight_scale_width [expr $SCALE_DATATYPE_WIDTH*$OC_DIMENSION]

    set weight_scale_double_buffer "DoubleBuffer<$weight_scale_width,$WEIGHT_SCALE_BUFFER_SIZE>"
    directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem0Run/mem0Run/mem0 -WORD_WIDTH $weight_scale_width
    directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem1Run/mem1Run/mem1 -WORD_WIDTH $weight_scale_width

    if {$TECHNOLOGY == "intel16"} {
      directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem0Run/mem0Run/mem0:rsc -MAP_TO_MODULE "intel16_32x256b_rf_wrapper_1r1w.intel16_32x256b_rf_wrapper_1r1w"
      directive set /Accelerator/$weight_scale_double_buffer/$weight_scale_double_buffer:mem1Run/mem1Run/mem1:rsc -MAP_TO_MODULE "intel16_32x256b_rf_wrapper_1r1w.intel16_32x256b_rf_wrapper_1r1w"
    }

    # Input scale buffer
    set INPUT_SCALE_BUFFER_SIZE [expr {$INPUT_BUFFER_SIZE / 32}]
    set input_scale_width [expr $SCALE_DATATYPE_WIDTH]

    set input_scale_double_buffer "DoubleBuffer<$input_scale_width,$INPUT_SCALE_BUFFER_SIZE>"

    if {$TECHNOLOGY == "intel16"} {
      directive set /Accelerator/$input_scale_double_buffer/$input_scale_double_buffer:mem0Run/mem0Run/mem0:rsc -MAP_TO_MODULE "intel16_2048x8b_rf_wrapper_1r1w.intel16_2048x8b_rf_wrapper_1r1w" 
      directive set /Accelerator/$input_scale_double_buffer/$input_scale_double_buffer:mem1Run/mem1Run/mem1:rsc -MAP_TO_MODULE "intel16_2048x8b_rf_wrapper_1r1w.intel16_2048x8b_rf_wrapper_1r1w"  
    }
  }
}
