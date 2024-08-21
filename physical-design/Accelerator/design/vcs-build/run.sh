set -o xtrace

# The last slash is required!!
DATA_DIR=${data_dir}/${network}/

if [ "${sim_level}" == "rtl" ]; then
  VLOG_NAME=concat_sim_rtl.v
  EXTRA_FLAGS="STAGE=rtl"
else
  VLOG_NAME=${sim_level}.v
  EXTRA_FLAGS="GATE_VLOG_DEP=./${VLOG_NAME}/${VLOG_NAME}.vts STAGE=gate"
fi

VCS_FLAGS="-sverilog \
+v2k \
-timescale=1ns/10ps \
+define+INTC_NO_PWR_PINS \
+define+INTCNOPWR \
+define+INTEL_NO_PWR_PINS \
+define+INTC_MEM_FAST_SIM \
+define+INTEL_EMULATION \
+define+FUNCTIONAL \
+define+no_unit_delay \
"


# put all .v files into a single file and copy over to build dir
# exclude build directory and .stamp files
cat $(find -L inputs \( -path "inputs/build" -prune \) -o \( -name "*.v" -not -name ".*" -print \)) > ${VLOG_NAME}

# copy over the netlist
ln -fs $(realpath ${VLOG_NAME}) inputs/build/Catapult/Accelerator/Accelerator.v1/${VLOG_NAME}

pushd inputs/build/Catapult/Accelerator/Accelerator.v1
# build simulator
make -f scverify/Verify_concat_sim_rtl_v_vcs.mk build NETWORK=${network} TESTS=${layer} DATA_DIR=${DATA_DIR} SIMS=${sims} CLOCK_PERIOD=${clock_period} NETLIST_LEAF=${sim_level} VCS_VLOGAN_OPTS="${VCS_FLAGS}" ${EXTRA_FLAGS}
popd

ln -s ../inputs/build outputs/build
