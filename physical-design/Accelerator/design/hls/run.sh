set -o xtrace

CATAPULT_BUILD_DIR=build/Catapult
mkdir -p ${CATAPULT_BUILD_DIR}

if [[ "${technology}" == "intel16" ]]; then
  ## Generate memory models
  for mem in $(ls inputs/*.v); do
    # Add these directives to memory models so that catapult doesn't complain
    sed '1i\
`define INTC_EMULATION' \
      $mem >memories/$(basename $mem)
    # WARN: We may want to remove this define at HLS so that we can simulate without this flag. But so far we don't do that

    # Add translate off directive
    sed -i '1i// synopsys translate_off' memories/$(basename $mem)
    sed -i '$a// synopsys translate_on' memories/$(basename $mem)
  done

  for tcl in $(ls memories/intel16_sram*.tcl); do
    catapult -shell -file $tcl
  done
fi

if [ "${waveform}" == "True" ]; then
  # build/Catapult will be the root directory of vcs simulations
  echo "dump -add sc_main/harness/accelerator/ccs_rtl" >build/Catapult/dump.do
fi

# generate RTL
make -j8 Accelerator BUILD_DIR=build CATAPULT_BUILD_DIR=${CATAPULT_BUILD_DIR} DATATYPE=${datatype} OC_DIMENSION=${oc_dimension} IC_DIMENSION=${ic_dimension} INPUT_BUFFER_SIZE=${input_buffer_size} WEIGHT_BUFFER_SIZE=${weight_buffer_size} ACCUM_BUFFER_SIZE=${accum_buffer_size} CLOCK_PERIOD=${clock_period} TECHNOLOGY=${technology}

# WARN: so far concat_rtl.v and concat_sim_rtl.v are identical, but may need to differentiate for synthesis and simulation
[[ -f ${CATAPULT_BUILD_DIR}/Accelerator/Accelerator.v1/concat_rtl.v ]] || {
  echo "Error: RTL generation failed"
  exit 1
}

cd outputs
# Must link over Accelerator level and no name change for scverify makefile to work
ln -s ../build

cp build/Catapult/Accelerator/Accelerator.v1/concat_rtl.v design.v
cp build/Catapult/Accelerator/Accelerator.v1/concat_sim_rtl.v design.sim.v

# Renaming modules
OLD_IFS=$IFS
IFS=" "

for vname in design.v design.sim.v; do
  modname=$(grep -oP "(?<=module )ProcessingElement\w*" $vname | tail -1)
  sed -i "s/\<$modname\>/ProcessingElement/g" $vname
  modname=$(grep -oP "(?<=module )SystolicArray\w*" $vname | tail -1)
  sed -i "s/\<$modname\>/SystolicArray/g" $vname
  modname=$(grep -oP "(?<=module )VectorUnit\w*" $vname | tail -1)
  sed -i "s/\<$modname\>/VectorUnit/g" $vname
done

IFS=$OLD_IFS

cd ..

