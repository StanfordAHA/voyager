set -o xtrace

CATAPULT_BUILD_DIR=build/Catapult
mkdir -p ${CATAPULT_BUILD_DIR}
if [ "${waveform}" == "True" ]; then
  # build/Catapult will be the root directory of vcs simulations
  echo "dump -add sc_main/harness/accelerator/ccs_rtl" >build/Catapult/dump.do
fi

# build TestRunner
make -j TestRunner BUILD_DIR=build CATAPULT_BUILD_DIR=${CATAPULT_BUILD_DIR} DATATYPE=${datatype} IC_DIMENSION=${ic_dimension} OC_DIMENSION=${oc_dimension}
# generate RTL
# note: make rtl would try to write to release folder as well, so I bypass it this way. Ugly, but works for now
make -j8 ${CATAPULT_BUILD_DIR}/Accelerator/Accelerator.v1/concat_rtl.v BUILD_DIR=build CATAPULT_BUILD_DIR=${CATAPULT_BUILD_DIR} DATATYPE=${datatype} OC_DIMENSION=${oc_dimension} IC_DIMENSION=${ic_dimension} CLOCK_PERIOD=${clock_period} TECHNOLOGY=${technology}

# WARN: so far concat_rtl.v and concat_sim_rtl.v are identical, but may need to differentiate for synthesis and simulation
[[ -f ${CATAPULT_BUILD_DIR}/Accelerator/Accelerator.v1/concat_rtl.v ]] || {
  echo "Error: RTL generation failed"
  exit 1
}

cd outputs
# Must link over Accelerator level and no name change for scverify makefile to work
ln -s ../build

# Uncomment the intel16 sram
awk '
  /\/\*/ { comment = 1 }
  /\*\// { comment = 1 }
  
  # Once start_marker is found, set the flag
  /module intel16_1024x.*_rf_wrapper/ { within_block = 1 }

  # print if meets the criteria
  !within_block || within_block && !comment { print }
  # within_block && !comment { print }
  
  # Mark it off once the end_marker is found
  within_block && /endmodule/ { within_block = 0 }

  # Reset the comment flag in every line
  comment = 0

' build/Catapult/Accelerator/Accelerator.v1/concat_rtl.v > design.v

# Renaming modules
modname=$(grep -oP "(?<=module )ProcessingElement.*$" design.v | tail -1)
sed -i "s/\<$modname\>/ProcessingElement/g" design.v
modname=$(grep -oP "(?<=module )SystolicArray.*$" design.v | tail -1)
sed -i "s/\<$modname\>/SystolicArray/g" design.v

cd ..

