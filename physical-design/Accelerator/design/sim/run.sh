set -o xtrace

EXTRA_FLAGS=""
if [ "${sim_level}" == "syn" ]; then
  VLOG_NAME=${sim_level}.v
  EXTRA_FLAGS="GATE_VLOG_DEP=./${VLOG_NAME}/${VLOG_NAME}.vts STAGE=gate"
fi

if [ "${waveform}" == "True" ]; then
  fsdb_name=${sim_level}_${network}_${layer}
  VCS_VCSSIM_OPTS="+fsdbfile+${fsdb_name}.fsdb +fsdb+all=on +fsdb+dumpon+0"
fi

export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
export PROJECT_ROOT=$(pwd)

pushd inputs/build/Catapult/Accelerator/Accelerator.v1
# run sim
make -f scverify/Verify_concat_sim_rtl_v_vcs.mk sim CODEGEN_DIR=test/compiler NETWORK=${network} TESTS=${layer} DATATYPE=${datatype} SIMS=${sims} CLOCK_PERIOD=${clock_period} NETLIST_LEAF=${sim_level} VCS_VCSSIM_OPTS="${VCS_VCSSIM_OPTS}" ${EXTRA_FLAGS}
popd

# Convert fsdb to saif
if [[ "${waveform}" == "True" && "${gen_saif}" == "True" ]]; then
  fsdb2saif inputs/build/Catapult/${fsdb_name}.fsdb -o outputs/run.saif
fi

# Save disk space
rm -f inputs/build/Catapult/${fsdb_name}.fsdb
